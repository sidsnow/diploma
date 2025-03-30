## train ip adapter
create data json from folder with images. run training. to resume training from ip adapter add `--pretrained_ip_adapter_path=path_to_adapter.bin"`

```bash
cd src
python create_data_json.py --folder_path folder --output_file output_json_file.json
accelerate launch --num_processes 8  --mixed_precision "fp16" tutorial_train.py --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" --image_encoder_path="models\\image_encoder" --data_json_file=".\\output_json_file.json" --data_root_path=".\\" --resolution=512 --train_batch_size=4 --dataloader_num_workers=4 --learning_rate=1e-04  --weight_decay=0.01  --output_dir="your_dir_for_checkpoints"   --save_steps=1000 --mixed_precision fp16
```