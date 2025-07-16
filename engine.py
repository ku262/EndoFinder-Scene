import torch
import json
import os
import numpy as np
from EndoFinder.tools.retrieval import evaluate_inference

def inference(args, val_loader, model, device, mean=False, cls=False):

    model.eval()

    query_jsonl, ref_jsonl, images_jsonl = [], [], []

    for batch_idx, samples in enumerate(val_loader):

        if cls:
            B, N = samples['images'].shape[:2]
            input_images = samples['images'].to(device, non_blocking=True)
        else:
            B, N_q = samples['query'].shape[:2]
            N_r = samples['ref'].shape[1]
            query, ref = samples['query'].to(device, non_blocking=True), samples['ref'].to(device, non_blocking=True)

        with torch.no_grad():
            if cls:
                if mean:
                    input_images = input_images.flatten(0, 1)
                    input_outputs = model(input_images)
                    input_outputs = input_outputs.view(B, N, -1)
                    input_scene_emb = torch.mean(input_outputs, dim=1)
                    # input_scene_emb = input_outputs[:, 0, :]
                else:
                    input_outputs = model.forward_inference(input_images)
                    input_scene_emb = input_outputs
            else:
                if mean:
                    query = query.flatten(0, 1)
                    ref = ref.flatten(0, 1)
                    query_outputs = model(query)
                    ref_outputs = model(ref)
                    query_outputs = query_outputs.view(B, N_q, -1)
                    ref_outputs = ref_outputs.view(B, N_r, -1)

                    query_scene_emb = torch.mean(query_outputs, dim=1)
                    ref_scene_emb = torch.mean(ref_outputs, dim=1)
                    # query_scene_emb = query_outputs[:,0,:]
                    # ref_scene_emb = ref_outputs[:,0,:]
                else:
                    query_outputs = model.forward_inference(query)
                    ref_outputs = model.forward_inference(ref)
                    query_scene_emb = query_outputs
                    ref_scene_emb = ref_outputs

        if cls:
            for i in range(B):
                data = {}
                data['scene_path'] = samples['scene_path'][i]
                data['img_names'] = list(np.array(samples['img_paths']).T[i])
                data['embedding'] = input_scene_emb[i].detach().cpu().numpy().tolist()
                data['ground_truth'] = int(samples['ground_truth'][i].detach().cpu().numpy())
                images_jsonl.append(data)
        else:
            for i in range(B):
                query_data = {}
                query_data['path'] = samples['query_path'][i]
                query_data['img_names'] = list(np.array(samples['query_img_paths']).T[i])
                query_data['embedding'] = query_scene_emb[i].detach().cpu().numpy().tolist()
                query_jsonl.append(query_data)
                ref_data = {}
                ref_data['path'] = samples['ref_path'][i]
                ref_data['img_names'] = list(np.array(samples['ref_img_paths']).T[i])
                ref_data['embedding'] = ref_scene_emb[i].detach().cpu().numpy().tolist()
                ref_jsonl.append(ref_data)

    if not os.path.exists(args.jsonl_path):
        os.makedirs(args.jsonl_path)

    if cls:
        with open(os.path.join(args.jsonl_path, "val.jsonl"), 'w', encoding='utf-8') as f:
            for item in images_jsonl:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')  
    else:
        with open(os.path.join(args.jsonl_path, "query.jsonl"), 'w', encoding='utf-8') as f:
            for item in query_jsonl:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')     
        with open(os.path.join(args.jsonl_path, "ref.jsonl"), 'w', encoding='utf-8') as f:
            for item in ref_jsonl:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')        

def train_one_epoch(args, epoch, model, data_loader, device, 
                    optimizer, scheduler, log_writer=None):

    model.train()

    # latent_scenes, instance_idss, targets, preds, masks = 0, 0, 0, 0, 0
    for data_iter_step, samples in enumerate(data_loader):

        target_image = samples['target_image']
        input_images = samples['input_images']
        instance_ids = samples["instance_id"]

        input_images = input_images.to(device, non_blocking=True)
        target_image = target_image.to(device, non_blocking=True)
        instance_ids = instance_ids.to(device, non_blocking=True)

        optimizer.zero_grad()
        
        mse_loss, _, _, sscd_loss, sscd_stats = model(input_images, target_image, instance_ids, mask_ratio=args.mask_ratio, entropy_weight=args.entropy_weight)
        
        loss = args.mse_weight * mse_loss + args.sscd_weight * sscd_loss


        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch}, [{data_iter_step}/{len(data_loader)} loss: {loss_value}, lr: {lr}] ")

        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('positive_sim', sscd_stats["positive_sim"].item(), epoch_1000x)
            log_writer.add_scalar('negative_sim', sscd_stats["negative_sim"].item(), epoch_1000x)
            log_writer.add_scalar('nearest_negative_sim', sscd_stats["nearest_negative_sim"].item(), epoch_1000x)
            log_writer.add_scalar('InfoNCE', sscd_stats["InfoNCE"].item(), epoch_1000x)
            log_writer.add_scalar('entropy', sscd_stats["entropy"].item(), epoch_1000x)
            log_writer.add_scalar('mae_loss', mse_loss.item(), epoch_1000x)
            log_writer.add_scalar('sscd_loss', sscd_loss.item(), epoch_1000x)

    return loss_value

def evaluate(args, epoch, val_loader, model, device, log_writer=None):

    model.eval()

    query_embeddings = []
    ref_embeddings = []
    query_names = []
    ref_names = []

    for batch_idx, samples in enumerate(val_loader):

        B, N_q = samples['query'].shape[:2]
        N_r = samples['ref'].shape[1]
        query, ref = samples['query'].to(device, non_blocking=True), samples['ref'].to(device, non_blocking=True)

        with torch.no_grad():

            query_outputs = model.forward_inference(query)
            ref_outputs = model.forward_inference(ref)
            query_scene_emb = query_outputs
            ref_scene_emb = ref_outputs

        for i in range(B):
            query_embeddings.append(query_scene_emb[i].detach().cpu().numpy().tolist())
            ref_embeddings.append(ref_scene_emb[i].detach().cpu().numpy().tolist())
            query_names.append(samples['query_path'][i])
            ref_names.append(samples['ref_path'][i])

    metrics = evaluate_inference(query_embeddings, ref_embeddings, query_names, ref_names, mode=args.mode)
    if log_writer is not None:
        """ We use epoch_1000x as the x-axis in tensorboard.
        This calibrates different curves when batch size changes.
        """
        log_writer.add_scalar('uAP', metrics['uAP'], epoch)
        log_writer.add_scalar('accuracy-at-1', metrics['accuracy-at-1'], epoch)
        log_writer.add_scalar('recall-at-p90', metrics['recall-at-p90'], epoch)

    print(f"Eval: [{metrics}] ")

    return metrics

def test(args, test_loader, model, device):

    model.eval()

    query_embeddings = []
    ref_embeddings = []
    query_names = []
    ref_names = []

    for batch_idx, samples in enumerate(test_loader):

        B, N_q = samples['query'].shape[:2]
        N_r = samples['ref'].shape[1]
        query, ref = samples['query'].to(device, non_blocking=True), samples['ref'].to(device, non_blocking=True)

        with torch.no_grad():
            
            # query = query.flatten(0, 1)
            # ref = ref.flatten(0, 1)
            from torch.cuda.amp import autocast
            with autocast():

                query_outputs = model(query)
                ref_outputs = model(ref)
            # query_outputs = query_outputs.view(B, N_q, -1)
            # ref_outputs = ref_outputs.view(B, N_r, -1)

            # query_scene_emb = torch.mean(query_outputs, dim=1)
            # ref_scene_emb = torch.mean(ref_outputs, dim=1)
            # query_scene_emb = query_outputs[:,1,:]
            # ref_scene_emb = ref_outputs[:,1,:]
            query_scene_emb = query_outputs
            ref_scene_emb = ref_outputs

        for i in range(B):
            query_embeddings.append(query_scene_emb[i, :].detach().cpu().numpy().tolist())
            ref_embeddings.append(ref_scene_emb[i, :].detach().cpu().numpy().tolist())
            query_names.append(samples['query_path'][i])
            ref_names.append(samples['ref_path'][i])


    metrics = evaluate_inference(query_embeddings, ref_embeddings, query_names, ref_names)
    print(f"Eval: [{metrics}] ")

    return metrics