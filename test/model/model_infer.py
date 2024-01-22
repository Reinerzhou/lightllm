import numpy as np
from multiprocessing import Queue
import multiprocessing

def test_model_inference(world_size, model_dir, model_class, batch_size, input_len, output_len, mode):
    ans_queue = Queue()
    model_kvargs = {
        "tp_rank": 0,
        "world_size": world_size,
        "weight_dir": model_dir,
        "max_total_token_num":batch_size * (input_len + output_len),
        "load_way": "HF",
        "mode": mode,
        "max_req_num": batch_size,
        "max_seq_length": (input_len + output_len)
    }

    tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue)
    return
    # workers = []
    # for rank_id in range(world_size):
    #     model_kvargs = {
    #         "tp_rank": rank_id,
    #         "world_size": world_size,
    #         "weight_dir": model_dir,
    #         "max_total_token_num":batch_size * (input_len + output_len),
    #         "load_way": "HF",
    #         "mode": mode,
    #         "max_req_num": batch_size,
    #         "max_seq_length": (input_len + output_len)
    #     }
    
    #     proc = multiprocessing.Process(target=tppart_model_infer, args=(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue))
    #     proc.start()
    #     workers.append(proc)

    # for proc in workers:
    #     proc.join()

    # assert not ans_queue.empty()
    # while not ans_queue.empty():
    #     assert ans_queue.get()
    # return 


def tppart_model_infer(model_class, model_kvargs, batch_size, input_len, output_len, ans_queue):
    import torch
    import torch.distributed as dist
    rank_id = model_kvargs["tp_rank"]
    world_size = model_kvargs["world_size"]

    dist.init_process_group('nccl', init_method='tcp://127.0.0.1:28765', rank=rank_id, world_size=world_size)
    torch.cuda.set_device(rank_id)

    import torch.distributed as dist
    dist.barrier()
    torch.cuda.empty_cache()

    max_prompt_size = 32
    max_seq_len = 64

    model_part = model_class(model_kvargs)
    
    total_len = min(max_seq_len, max_prompt_size + output_len)

    test_data = np.vstack([np.arange(5, total_len + 5) for _ in range(batch_size)])
    test_data = torch.from_numpy(test_data).cuda()

    left_pad_size_list = []
    for k, t in enumerate(test_data):
        left_pad_size = output_len - len(t)
        left_pad_size_list.append(left_pad_size)
        test_data[k, left_pad_size: output_len] = torch.tensor(t).cuda().long()
        if left_pad_size > 0:
            test_data[k, 0: left_pad_size] = torch.full((1, left_pad_size), 0).cuda().long()

    test_data = test_data.reshape(-1)

    # input_text_mask = test_data != -1

    b_req_idx = model_part.req_manager.alloc(batch_size).int()
    b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        b_start_loc[i] = i * max_prompt_size
        b_seq_len[i] = max_prompt_size

    start_pos = max_prompt_size
    prev_pos = 0

    for cur_pos in range(start_pos, total_len):
        seqlen = cur_pos - prev_pos
        if seqlen > 1:
            origin_mask_full = torch.zeros((1, 1, seqlen, seqlen), 
                                           dtype=torch.float16, device="cuda")
            mask_list = []
            for pad_size in left_pad_size_list:
                right_corner_mask = torch.full((1, 1, seqlen - pad_size, seqlen - pad_size),
                                                float("-inf"), device="cuda")
                right_corner_mask = torch.triu(right_corner_mask, diagonal=prev_pos + 1).to(torch.float16)
                final_mask_full = origin_mask_full.clone()
                left_corner_mask = torch.full((1, 1, seqlen - pad_size, pad_size), 
                                              float("-inf"), device="cuda").to(torch.float16)
                final_mask_full[:, :, pad_size:, pad_size:] = right_corner_mask
                final_mask_full[:, :, pad_size:, :pad_size] = left_corner_mask
                mask_list.append(final_mask_full)
            mask = torch.cat(mask_list, dim=0)

            total_token_num = seqlen * batch_size

            logics = model_part.forward(batch_size, 
                                        total_token_num, 
                                        seqlen, 
                                        test_data,
                                        mask,
                                        b_req_idx,
                                        b_start_loc,
                                        b_seq_len,
                                        is_prefill=True)
            
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()
        else:
            origin_mask_full = torch.zeros((1, 1, seqlen, cur_pos),
                                            dtype=torch.float16, device="cuda")
            mask_list = []
            for pad_size in left_pad_size_list:
                left_corner_mask = torch.full((1, 1, seqlen, pad_size), float("-inf"),
                                                device="cuda").to(torch.float16)
                final_mask_full = origin_mask_full.clone()
                final_mask_full[:, :, :, :pad_size] = left_corner_mask
                mask_list.append(final_mask_full)
            mask = torch.cat(mask_list, dim=0)

            b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
            total_token_num += batch_size
            b_seq_len += 1

            logics = model_part.forward(batch_size, total_token_num, cur_pos, torch.from_numpy(
                predict_ids).cuda().reshape(-1), mask, b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
            
            prob_out = torch.softmax(logics, dim=-1)
            predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
            predict_ids = predict_ids.detach().cpu().numpy()

    # warm up
    # test_data = np.vstack([np.arange(5, input_len + 5) for _ in range(batch_size)])
    # test_data = test_data.reshape(-1)
    # test_data = torch.from_numpy(test_data).cuda()

    # b_req_idx = model_part.req_manager.alloc(batch_size).int()
    # b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    # b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    # for i in range(batch_size):
    #     b_start_loc[i] = i * input_len
    #     b_seq_len[i] = input_len

    # total_token_num = input_len * batch_size
    # logics = model_part.forward(batch_size, 
    #                             total_token_num, 
    #                             input_len, 
    #                             test_data,
    #                             b_req_idx,
    #                             b_start_loc,
    #                             b_seq_len,
    #                             is_prefill=True)
    # prob_out = torch.softmax(logics, dim=-1)
    # predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    # predict_ids = predict_ids.detach().cpu().numpy()

    # for i in range(output_len):
    #     b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
    #     total_token_num += batch_size
    #     b_seq_len += 1
    #     logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
    #         predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
    #     prob_out = torch.softmax(logics, dim=-1)
    #     predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    #     predict_ids = predict_ids.detach().cpu().numpy()

    model_part.mem_manager.free_all()
    model_part.req_manager.free_all()
    
    if rank_id == 0:
        print("can use mem size:", model_part.mem_manager.can_use_mem_size)
        print("can use req size:", model_part.req_manager.can_use_req_size)
        
    b_req_idx = None
    b_start_loc = None
    b_seq_len = None
    
    dist.barrier()
    import time
    torch.cuda.synchronize()
    # start_time = time.time()

    # prefill_start_time = time.time()

    # b_req_idx = model_part.req_manager.alloc(batch_size).int()
    # b_start_loc = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    # b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    # for i in range(batch_size):
    #     b_start_loc[i] = i * input_len
    #     b_seq_len[i] = input_len

    # total_token_num = batch_size * input_len
    # logics = model_part.forward(batch_size, total_token_num, input_len, test_data,
    #                                              b_req_idx, b_start_loc, b_seq_len, is_prefill=True)
    # prob_out = torch.softmax(logics, dim=-1)
    # predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    # predict_ids = predict_ids.detach().cpu().numpy()

    # torch.cuda.synchronize()
    # if rank_id == 0:
    #     print("prefill time cost:", (time.time() - prefill_start_time) * 1000)

    # for i in range(output_len):
    #     torch.cuda.synchronize()
    #     step_start = time.time()
    #     b_start_loc = b_start_loc + torch.arange(0, batch_size, dtype=torch.int32, device="cuda")
    #     total_token_num += batch_size
    #     b_seq_len += 1

    #     logics = model_part.forward(batch_size, total_token_num, input_len + i + 1, torch.from_numpy(
    #         predict_ids).cuda().reshape(-1), b_req_idx, b_start_loc, b_seq_len, is_prefill=False)
    #     prob_out = torch.softmax(logics, dim=-1)
    #     predict_ids = torch.argmax(prob_out, dim=1, keepdim=True)
    #     predict_ids = predict_ids.detach().cpu().numpy()
    #     torch.cuda.synchronize()
    #     if i % 100 == 0 or i == output_len - 1:
    #         if rank_id == 0:
    #             print(i, "step cost time:", (time.time() - step_start) * 1000)

    # torch.cuda.synchronize()
    # end_time = time.time()

    # if rank_id == 0:
    #     print("time total cost(ms):", (end_time - start_time) * 1000)
    ans_queue.put(True)

    return


