import math
import torch
from typing import List, Dict
from torch.optim import Optimizer
from torch.optim import lr_scheduler


def get_lr_scheduler(
    optimizer: Optimizer,
    **kwargs
):
    """
    根据 kwargs['scheduler_type'] 返回对应的学习率调度器 (lr scheduler)。

    支持的 scheduler_type：
    - "step"                : StepLR，按固定间隔阶梯式衰减
    - "multi_step"          : MultiStepLR，在指定 epoch 列表处阶梯式衰减
    - "exponential"         : ExponentialLR，按固定比例每个 step/epoch 指数衰减
    - "cosine"              : CosineAnnealingLR，余弦退火
    - "cosine_warm_restart" : CosineAnnealingWarmRestarts，带周期性重启的余弦退火
    - "reduce_on_plateau"   : ReduceLROnPlateau，指标长期不提升时降低 lr
    - "cosine_warmup"       : 先线性预热，再余弦退火（自定义 LambdaLR）
    - "custom_multi_step"   : 多步自定义，指定到某些 epoch 时 lr 变为某个绝对值
    - "const"               : 保持不变
    """

    scheduler_type = kwargs['scheduler_type'].lower()

    # 1. StepLR：固定间隔阶梯衰减
    if scheduler_type == "step":
        """
        调度方式：
        - 每隔 step_size 个 step/epoch，将学习率乘以 gamma：
          lr_t = lr_0 * (gamma ** floor(t / step_size))

        参数：
        - step_size (int)  ：衰减间隔
        - gamma (float)    ：每次衰减倍率，默认 0.5
        """
        step_size: int = kwargs.get("step_size", 30)
        gamma: float = kwargs.get("gamma", 0.5)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 2. MultiStepLR：在多处阶梯衰减
    elif scheduler_type == "multi_step":
        """
        调度方式：
        - 在 milestones 指定的 epoch/step 上，将 lr 乘以 gamma（可以多次）

        参数：
        - milestones (List[int])：衰减的 epoch/step 列表（必须提供）
        - gamma (float)         ：每次衰减倍率，默认 0.5
        """
        milestones: List[int] = kwargs["milestones"]
        gamma: float = kwargs.get("gamma", 0.5)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # 3. ExponentialLR：指数衰减
    elif scheduler_type == "exponential":
        """
        调度方式：
        - 每次调用 scheduler.step() 时，将 lr 乘以 gamma：
          lr_t = lr_0 * (gamma ** t)

        参数：
        - gamma (float)：衰减因子，0<gamma<1 时衰减
        """
        gamma: float = kwargs.get("gamma", 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # 4. CosineAnnealingLR：余弦退火
    elif scheduler_type == "cosine":
        """
        调度方式：
        - 在 [0, T_max] 内按照余弦函数从初始 lr 平滑下降到 eta_min：
          lr_t = eta_min + (lr_0 - eta_min) * (1 + cos(pi * t / T_max)) / 2

        参数：
        - T_max (int)    ：一个完整余弦周期的长度（通常是总 epoch 数）
        - eta_min (float)：最小学习率，默认 0.0
        """
        T_max: int = kwargs.get("T_max", 50)
        eta_min: float = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    # 5. CosineAnnealingWarmRestarts：带重启的余弦退火
    elif scheduler_type == "cosine_warm_restart":
        """
        调度方式：
        - 使用余弦退火，但会周期性地重启；
        - 周期长度从 T_0 开始，每次乘以 T_mult；
        - 每个周期内从 lr_0 退火到 eta_min，然后重置回 lr_0 再退火。

        参数：
        - T_0 (int)     ：第一个周期长度
        - T_mult (int)  ：每次重启后周期长度放大倍数，默认 2
        - eta_min (float)：每个周期的最低 lr，默认 0.0
        """
        T_0: int = kwargs.get("T_0", 10)
        T_mult: int = kwargs.get("T_mult", 2)
        eta_min: float = kwargs.get("eta_min", 0.0)
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )

    # 6. ReduceLROnPlateau：指标停滞时降低学习率
    elif scheduler_type == "reduce_on_plateau":
        """
        调度方式：
        - 根据监控指标（如 val_loss）变化情况来调整 lr；
        - 当指标在 patience 个 epoch 内没有明显改善时，将 lr 乘以 factor；
        - 使用方式：每个 epoch 结束时调用 scheduler.step(metric_value)。

        参数（常用）：
        - mode (str)      ："min" 或 "max"，默认 "min"
        - factor (float)  ：每次降低 lr 的倍率，如 0.5
        - patience (int)  ：容忍多少个 epoch 不提升
        - threshold (float)：认为“有提升”的最小变化
        - min_lr (float)  ：lr 下界
        """
        mode: str = kwargs.get("mode", "min")
        factor: float = kwargs.get("factor", 0.5)
        patience: int = kwargs.get("patience", 10)
        threshold: float = kwargs.get("threshold", 1e-4)
        min_lr: float = kwargs.get("min_lr", 0.0)

        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
        )

    # 7. 自定义：预热 + 余弦退火 (cosine_warmup)
    elif scheduler_type == "cosine_warmup":
        """
        调度方式：
        - 阶段 1：预热 (warmup)
            前 warmup_epochs 个 epoch 内，
            lr 从 warmup_start_lr 线性上升到 base_lr（optimizer 当前 lr）。
        - 阶段 2：余弦退火 (cosine)
            之后的 (max_epochs - warmup_epochs) 个 epoch 内，
            使用余弦退火从 base_lr 下降到 eta_min。

        参数：
        - warmup_epochs (int)    ：预热 epoch 数
        - max_epochs (int)       ：总 epoch 数（包含预热）
        - warmup_start_lr (float)：预热起始 lr，默认 0.0
        - eta_min (float)        ：余弦退火的最低 lr，默认 0.0

        用法：
        - 一般在每个 epoch 结束后调用 scheduler.step()。
        """
        warmup_epochs = kwargs.get("warmup_epochs", 5)
        max_epochs = kwargs.get("max_epochs", 100)
        eta_min = kwargs.get("eta_min", 0.0)

        # 假设所有 param_group 的 lr 相同
        base_lr = optimizer.param_groups[0]["lr"]
        warmup_start_lr = kwargs.get("warmup_start_lr", 0.0)

        def lr_lambda(current_epoch: int):
            # 阶段 1：线性预热
            if current_epoch < warmup_epochs:
                warmup_progress = current_epoch / max(1, warmup_epochs)
                lr = warmup_start_lr + (base_lr - warmup_start_lr) * warmup_progress
                return lr / base_lr  # 转成倍率

            # 阶段 2：余弦退火
            cos_epoch = current_epoch - warmup_epochs
            cos_total = max_epochs - warmup_epochs
            cos_total = max(1, cos_total)

            cos_factor = 0.5 * (1 + math.cos(math.pi * cos_epoch / cos_total))
            lr = eta_min + (base_lr - eta_min) * cos_factor
            return lr / base_lr

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 8. 自定义：多步自定义绝对 lr (custom_multi_step)
    elif scheduler_type == "custom_multi_step":
        """
        调度方式：
        - 用户指定一个字典：lr_milestones = {epoch: lr_value, ...}
        - 当 current_epoch >= 某个 epoch 时，lr 变为对应的 lr_value；
        - 如果有多个满足条件的 epoch，取“最大且不超过当前 epoch”的那个；
        - 例如 lr_milestones = {10: 0.01, 30: 0.001}：
            - 0 <= epoch < 10：lr = base_lr（optimizer 初始 lr）
            - 10 <= epoch < 30：lr = 0.01
            - epoch >= 30    ：lr = 0.001

        参数：
        - lr_milestones (Dict[int, float])：
            key   : epoch 编号（从 0 开始的整数）
            value : 该 epoch 及之后使用的绝对 lr 值

        注意：
        - 这是“绝对值调度”，不是按比例乘法；
        - 通过 LambdaLR 实现，需要用 base_lr 把绝对 lr 转成倍率。
        """
        lr_milestones: Dict[int, float] = kwargs["lr_milestones"]
        if not lr_milestones:
            raise ValueError("`custom_multi_step` requires non-empty `lr_milestones` dict.")

        # 初始 lr 作为 base_lr
        base_lr = optimizer.param_groups[0]["lr"]

        # 先把 milestone 的 epoch 排序，方便查找
        sorted_epochs = sorted(lr_milestones.keys())

        def lr_lambda(current_epoch: int):
            # 找到最后一个 <= current_epoch 的 milestone
            target_lr = base_lr  # 默认用初始 lr
            for e in sorted_epochs:
                if current_epoch >= e:
                    target_lr = lr_milestones[e]
                else:
                    break
            # 转成倍率
            return target_lr / base_lr

        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # 9. 自定义：保持不变 (const)
    elif scheduler_type == "const":
        """
        调度方式：
        - 保持 lr 不变，不进行任何调整。

        参数：
        - 无

        用法：
        - 一般在每个 epoch 结束后调用 scheduler.step()。
        """
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    else:
        raise ValueError(f"Unknown lr scheduler type: {scheduler_type}")