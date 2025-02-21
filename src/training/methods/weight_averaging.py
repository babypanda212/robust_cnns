
model_ema = torch.optim.swa_utils.AveragedModel(model_adv, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9))

model_adv = model_adv.to(device)
model_ema = model_ema.to(device)

