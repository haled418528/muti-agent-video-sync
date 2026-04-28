# Multi-Agent Video Lip-Sync

一个基于多 Agent 协作的视频口型同步训练系统。通过生成器（Generator）与评价器（Evaluator）的迭代循环，不断优化口型同步效果，最终达到目标质量。

## 核心架构

```
用户提交任务（视频 + 音频）
           ↓
      [Ralph 持久化循环]
           ↓
    ┌───────────────────────┐
    │   Agent A (Generator) │
    │   - 口型同步生成       │
    │   - 接收反馈修改提示词  │
    └───────────┬───────────┘
    ┌───────────┴───────────┐
    │   Agent B (Evaluator) │
    │   - AI自动化指标评分    │
    │   - 触发人工评判通道    │
    └───────────┬───────────┘
                ↓ 反馈
           [收敛判断]
                ↓
         达标 → 输出最终视频
```

## Agent 职责

### Agent A — Generator（生成器）

- 调用底层口型同步模型（Wav2Lip/DiffTV）
- 管理生成提示词模板
- 根据 B 的反馈修改提示词/参数重新生成
- 保存中间结果和版本历史

### Agent B — Evaluator（评价器）

- **AI 自动化评分**（每次生成后立即执行）:
  - SyncNet 距离: 口型与音频同步精度
  - LSE-D (Lip Sync Error - Distance): Wav2Lip 官方指标
  - LSE-C (Lip Sync Error - Confidence): 同步置信度
- **人工评判触发**: 当 AI 指标达标率 > 80%，生成人工评判样本
- **反馈生成**: 汇总 AI 指标 + 人工判定，生成结构化反馈给 A

### Ralph — 持久化循环控制

- 控制整个生成-评价-反馈循环
- 记录每次迭代的状态
- 判断收敛条件（达标 / 超过最大迭代次数 / 明确失败）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载预训练模型（首次运行自动下载）
python download_models.py

# 运行主程序
python main.py --video path/to/video.mp4 --audio path/to/audio.wav
```

## 配置

- `config/model_config.json` — 模型路径、参数配置
- `config/eval_thresholds.json` — 评价指标阈值
- `config/prompt_templates.json` — A 的提示词模板

## 项目结构

```
muti-agent-video-sync/
├── config/                 # 配置文件
├── src/
│   ├── generator/         # 生成器模块
│   ├── evaluator/         # 评价器模块
│   └── ralph/             # 循环控制器模块
├── workspace/             # 迭代输出目录
├── tests/                 # 测试用例
└── docs/                  # 文档
```

## 扩展接口

本项目预留了以下扩展点，方便后续添加新功能：

- `src/generator/base.py` — 定义 Generator 基类，可接入新的口型同步模型
- `src/evaluator/base.py` — 定义 Evaluator 基类，可添加新的评价指标
- `src/ralph/loop_controller.py` — 循环控制器，可自定义收敛判断逻辑
- `config/prompt_templates.json` — 提示词模板可动态扩展

## TODO

- [ ] 实现 Wav2Lip 口型同步生成模块
- [ ] 实现 SyncNet 指标计算
- [ ] 实现 Claude Code Team 集成
- [ ] 实现 Ralph 循环控制器
- [ ] 添加人工评判界面
- [ ] 支持 DiffTV 模型
- [ ] 支持批量处理

## License

MIT
