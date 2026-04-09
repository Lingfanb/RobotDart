# DART Training Commands (BABEL + SMPL-X)

## 0. Data Preparation
```bash
python -m data_scripts.extract_dataset
```

## 1. Train Motion Primitive VAE
```bash
python -m mld.train_mvae --track 1 --wandb_entity lingfanb-university-college-london-ucl- --exp_name 'mvae_babel_smplx' --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r8.yaml' --data_args.weight_scheme 'text_samp:0.' --train_args.batch_size 128 --train_args.weight_kl 1e-6 --train_args.stage1_steps 100000 --train_args.stage2_steps 50000 --train_args.stage3_steps 50000 --train_args.save_interval 50000 --train_args.weight_smpl_joints_rec 10.0 --train_args.weight_joints_consistency 10.0 --train_args.weight_transl_delta 100 --train_args.weight_joints_delta 100 --train_args.weight_orient_delta 100 --model_args.arch 'all_encoder' --train_args.ema_decay 0.999 --model_args.num_layers 7 --model_args.latent_dim 1 256
```

## 2. Train Latent Motion Primitive Diffusion Model
```bash
python -m mld.train_mld --track 1 --wandb_entity lingfanb-university-college-london-ucl- --exp_name 'mld_babel_smplx' --train_args.batch_size 1024 --train_args.use_amp 1 --data_args.dataset 'mp_seq_v2' --data_args.data_dir './data/seq_data_zero_male' --data_args.cfg_path './config_files/config_hydra/motion_primitive/mp_h2_f8_r4.yaml' --denoiser_args.mvae_path './mvae/mvae_babel_smplx/checkpoint_200000.pt' --denoiser_args.train_rollout_type 'full' --denoiser_args.train_rollout_history 'rollout' --train_args.stage1_steps 100000 --train_args.stage2_steps 100000 --train_args.stage3_steps 100000 --train_args.save_interval 100000 --train_args.weight_latent_rec 1.0 --train_args.weight_feature_rec 1.0 --train_args.weight_smpl_joints_rec 0 --train_args.weight_joints_consistency 0 --train_args.weight_transl_delta 1e4 --train_args.weight_joints_delta 1e4 --train_args.weight_orient_delta 1e4 --data_args.weight_scheme 'text_samp:0.' denoiser-args.model-args:denoiser-transformer-args
```

## 3. Train Motion Control Policy (Optional)
```bash
python -m control.train_reach_location_mld --track 1 --wandb_entity lingfanb-university-college-london-ucl- --exp_name 'control_policy' --denoiser_checkpoint './mld_denoiser/mld_fps_clip_euler/checkpoint_300000.pt' --total_timesteps 200000000 --env_args.export_interval 1000 --env_args.num_envs 256 --env_args.num_steps 32 --minibatch_size 1024 --update_epochs 10 --learning_rate 3e-4 --max_grad_norm 0.1 --env_args.texts 'walk' 'run' 'hop on left leg' --env_args.success_threshold 0.3 --env_args.weight_success 1.0 --env_args.weight_dist 1.0 --env_args.weight_foot_floor 100.0 --env_args.weight_skate 100.0 --env_args.weight_orient 0.1 --policy_args.min_log_std -1.0 --policy_args.max_log_std 1.0 --policy_args.latent_dim 512 --env_args.goal_dist_max_init 5.0 --env_args.goal_schedule_interval 50000 --policy_args.use_lora 0 --policy_args.lora_rank 16 --policy_args.n_blocks 2 --policy_args.use_tanh_scale 1 --policy_args.use_zero_init 1 --init_data_path './data/stand.pkl' --env_args.weight_rotation 10.0 --env_args.weight_delta 0.0 --env_args.obs_goal_angle_clip 60.0 --env_args.obs_goal_dist_clip 5.0 --env_args.use_predicted_joints 1 --env_args.goal_angle_init 120.0 --env_args.goal_angle_delta 0.0
```

## Notes
- Step 2 depends on Step 1's checkpoint (`mvae_babel_smplx/checkpoint_200000.pt`)
- Step 3's `--denoiser_checkpoint` path may need updating to match Step 2's output
- Uses [wandb](https://wandb.ai/) for logging — make sure you're logged in (`wandb login`)
