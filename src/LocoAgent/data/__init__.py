"""Teacher-rollout dataset for BC distillation.

Each rollout is a (T, 416) state + (T, 29) action trajectory collected from the PPO
tracker on LAFAN1 motions (walks + runs + sprints, 18 motions / ~1.4 h total).
"""
