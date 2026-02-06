# import os
# import wandb
# import gc
# from tqdm import tqdm
# import torch
# import json
# import pandas as pd
# from torch.utils.data import DataLoader
# from torch.nn.utils import clip_grad_norm_

# from src.model import load_model, llama_model_path
# from src.dataset import load_dataset
# from src.utils.evaluate import eval_funcs
# from src.config import parse_args_llama
# from src.utils.ckpt import _save_checkpoint, _reload_best_model
# from src.utils.collate import collate_fn
# from src.utils.seed import seed_everything
# from src.utils.lr_schedule import adjust_learning_rate


# def main(args):

#     # Step 1: Set up wandb
#     seed = args.seed
#     wandb.init(project=f"{args.project}",
#                mode="offline",
#                name=f"{args.dataset}_{args.model_name}_seed{seed}",
#                config=args)

#     seed_everything(seed=args.seed)
#     print(args)

#     dataset = load_dataset[args.dataset]()
#     idx_split = dataset.get_idx_split()

#     # Step 2: Build Node Classification Dataset
#     train_dataset = [dataset[i] for i in idx_split['train']]
#     val_dataset = [dataset[i] for i in idx_split['val']]
#     test_dataset = [dataset[i] for i in idx_split['test']]

#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

#     # Step 3: Build Model
#     args.llm_model_path = llama_model_path[args.llm_model_name]
#     model = load_model[args.model_name](graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt)

#     # Step 4 Set Optimizer
#     params = [p for _, p in model.named_parameters() if p.requires_grad]
#     optimizer = torch.optim.AdamW(
#         [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
#         betas=(0.9, 0.95)
#     )
#     trainable_params, all_param = model.print_trainable_params()
#     print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

#     # Step 5. Training
#     num_training_steps = args.num_epochs * len(train_loader)
#     progress_bar = tqdm(range(num_training_steps))
#     best_val_loss = float('inf')

#     for epoch in range(args.num_epochs):

#         model.train()
#         epoch_loss, accum_loss = 0., 0.

#         for step, batch in enumerate(train_loader):

            
#             optimizer.zero_grad()
#             loss = model(batch)
#             loss.backward()

#             clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

#             if (step + 1) % args.grad_steps == 0:
#                 adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

#             optimizer.step()
#             epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

#             if (step + 1) % args.grad_steps == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 wandb.log({'Lr': lr})
#                 wandb.log({'Accum Loss': accum_loss / args.grad_steps})
#                 accum_loss = 0.

#             progress_bar.update(1)

#         print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
#         wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

#         val_loss = 0.
#         eval_output = []
#         model.eval()
#         with torch.no_grad():
#             for step, batch in enumerate(val_loader):
#                 loss = model(batch)
#                 val_loss += loss.item()
#             val_loss = val_loss/len(val_loader)
#             print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
#             wandb.log({'Val Loss': val_loss})

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             _save_checkpoint(model, optimizer, epoch, args, is_best=True)
#             best_epoch = epoch

#         print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

#         if epoch - best_epoch >= args.patience:
#             print(f'Early stop at epoch {epoch}')
#             break

#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()

#     # Step 5. Evaluating
#     os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
#     path = f'{args.output_dir}/{args.dataset}/model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv'
#     print(f'path: {path}')

#     model = _reload_best_model(model, args)
#     model.eval()
#     progress_bar_test = tqdm(range(len(test_loader)))
#     with open(path, "w") as f:
#         for step, batch in enumerate(test_loader):
#             with torch.no_grad():
#                 output = model.inference(batch)
#                 df = pd.DataFrame(output)
#                 for _, row in df.iterrows():
#                     f.write(json.dumps(dict(row)) + "\n")
#             progress_bar_test.update(1)

#     # Step 6. Post-processing & compute metrics
#     acc = eval_funcs[args.dataset](path)
#     print(f'Test Acc {acc}')
#     wandb.log({'Test Acc': acc})


# if __name__ == "__main__":

#     args = parse_args_llama()

#     main(args)
#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     gc.collect()













#上面为可以运行的版本，但是一次只有一个结果，下面是20次的结果
import os
import wandb
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import load_model, llama_model_path
from src.dataset import load_dataset
from src.utils.evaluate import eval_funcs
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate
import csv

def main(args):

    # Step 1: Set up wandb
    seed = args.seed
    wandb.init(project=f"{args.project}",
               mode="offline",
               name=f"{args.dataset}_{args.model_name}_seed{seed}",
               config=args)

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    # Step 2: Build Node Classification Dataset
    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, 
                              pin_memory=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, 
                            pin_memory=True, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, 
                             pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Step 3: Build Model
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = load_model[args.model_name](
        graph_type=dataset.graph_type, args=args, init_prompt=dataset.prompt
    )

    # Step 4 Set Optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || "
          f"trainable%: {100 * trainable_params / all_param}")

    # Step 5. Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):

        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, 
                                     step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss += loss.item()
            accum_loss += loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb.log({'Lr': lr})
                wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: "
              f"Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

        # Validation
        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
        val_loss = val_loss / len(val_loader)
        print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")
        wandb.log({'Val Loss': val_loss})

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} '
              f'Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Step 5. Evaluating (load best checkpoint)
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = (f'{args.output_dir}/{args.dataset}/'
            f'model_name_{args.model_name}_llm_model_name_{args.llm_model_name}_'
            f'llm_frozen_{args.llm_frozen}_max_txt_len_{args.max_txt_len}_'
            f'max_new_tokens_{args.max_new_tokens}_gnn_model_name_{args.gnn_model_name}_'
            f'patience_{args.patience}_num_epochs_{args.num_epochs}_seed{seed}.csv')
    print(f'path: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))

    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    # Step 6. Post-processing & compute metrics
    acc = eval_funcs[args.dataset](path)
    print(f'Test Acc {acc}')
    wandb.log({'Test Acc': acc})

    return acc  # 返回测试集准确率，方便外部统计


# if __name__ == "__main__":

#     args = parse_args_llama()

#     # 连续训练 20 次，并把结果保存到一个列表里
#     all_acc = []
#     num_runs = 20
    
#     for run_idx in range(num_runs):
#         # 你可以选择：每次跑换一个 seed，或者固定 seed
#         # 如果想让 seed 每次都不同，可以加:
#         new_seed = args.seed + run_idx
#         args.seed = new_seed
        
#         print(f"\n=============================")
#         print(f"  Run {run_idx+1}/{num_runs}, seed={args.seed}")
#         print(f"=============================\n")

#         acc = main(args)
#         all_acc.append(acc)

#     # 统计每次结果
#     avg_acc = sum(all_acc) / len(all_acc)
#     print("\n=============================")
#     print(f"所有 {num_runs} 次运行的 Test Acc 依次为:")
#     print(all_acc)
#     print(f"平均 Test Acc: {avg_acc}")
#     print("=============================\n")

#     torch.cuda.empty_cache()
#     torch.cuda.reset_max_memory_allocated()
#     gc.collect()


if __name__ == "__main__":
    args = parse_args_llama()

    # 连续训练 20 次，并把结果保存到一个列表里
    all_acc = []
    num_runs = 20

    for run_idx in range(num_runs):
        # 设置不同的 seed，确保每次训练的随机性不同
        new_seed = args.seed + run_idx
        args.seed = new_seed

        print(f"\n=============================")
        print(f"  Run {run_idx+1}/{num_runs}, seed={args.seed}")
        print(f"=============================\n")

        acc = main(args)
        all_acc.append(acc)

        # 清理资源
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

        # 结束当前的 wandb run，防止内存积累
        wandb.finish()

    # 统计每次结果
    avg_acc = sum(all_acc) / len(all_acc)
    print("\n=============================")
    print(f"所有 {num_runs} 次运行的 Test Acc 依次为:")
    print(all_acc)
    print(f"平均 Test Acc: {avg_acc}")
    print("=============================\n")

    # 保存结果到 CSV 文件
    csv_file = "test_accuracies.csv"
    try:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Run", "Seed", "Test Accuracy"])
            for run_idx, acc in enumerate(all_acc, 1):
                writer.writerow([run_idx, args.seed - num_runs + run_idx, acc])
        print(f"所有运行结果已保存到 {csv_file}")
    except Exception as e:
        print(f"保存 CSV 文件时出错: {e}")

    # 最终清理
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()


