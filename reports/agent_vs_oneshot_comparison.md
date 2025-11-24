# Agent vs. One-Shot Generation Comparison (Iris Dataset)

Generated on: 2025-11-20 17:36:00.050887

## Best Performance by System
Comparison of the best performing configuration for each system.

| System                            | configuration   |   mean |    std |   ci_lower |   ci_upper |   n_runs |
|:----------------------------------|:----------------|-------:|-------:|-----------:|-----------:|---------:|
| Gemini One-Shot (live_flash)      | minimal         | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (2.5_pro)         | minimal         | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (live_flash_lite) | minimal         | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | minimal         | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| ADK Agent (Standard)              | no_scaling      | 0.9778 | 0.0385 |     0.8822 |     1.0734 |        3 |
| Gemini One-Shot (live_pro)        | full            | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (2.5_flash)       | minimal         | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (live)            | no_scaling      | 0.9667 | 0.0333 |     0.8839 |     1.0495 |        3 |
| ADK Agent (Wrapper Test)          | no_scaling      | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |

## Detailed Results
All configurations for each system, sorted by performance.

| System                            | configuration          |   mean |    std |   ci_lower |   ci_upper |   n_runs |
|:----------------------------------|:-----------------------|-------:|-------:|-----------:|-----------:|---------:|
| ADK Agent (Standard)              | no_scaling             | 0.9778 | 0.0385 |     0.8822 |     1.0734 |        3 |
| ADK Agent (Standard)              | no_scaling             | 0.9778 | 0.0385 |     0.8822 |     1.0734 |        3 |
| ADK Agent (Standard)              | no_scaling             | 0.9778 | 0.0385 |     0.8822 |     1.0734 |        3 |
| ADK Agent (Standard)              | no_scaling             | 0.9778 | 0.0385 |     0.8822 |     1.0734 |        3 |
| ADK Agent (Standard)              | minimal                | 0.9667 | 0.0577 |     0.8232 |     1.1101 |        3 |
| ADK Agent (Standard)              | minimal                | 0.9667 | 0.0577 |     0.8232 |     1.1101 |        3 |
| ADK Agent (Standard)              | minimal                | 0.9667 | 0.0577 |     0.8232 |     1.1101 |        3 |
| ADK Agent (Standard)              | minimal                | 0.9667 | 0.0577 |     0.8232 |     1.1101 |        3 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9633 | 0.0284 |     0.9500 |     0.9766 |       20 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9633 | 0.0284 |     0.9500 |     0.9766 |       20 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9633 | 0.0284 |     0.9500 |     0.9766 |       20 |
| ADK Agent (Standard)              | minimal                | 0.9600 | 0.0256 |     0.9480 |     0.9720 |       20 |
| ADK Agent (Standard)              | minimal                | 0.9600 | 0.0256 |     0.9480 |     0.9720 |       20 |
| ADK Agent (Standard)              | minimal                | 0.9600 | 0.0256 |     0.9480 |     0.9720 |       20 |
| ADK Agent (Standard)              | no_scaling             | 0.9567 | 0.0244 |     0.9452 |     0.9681 |       20 |
| ADK Agent (Standard)              | no_scaling             | 0.9567 | 0.0244 |     0.9452 |     0.9681 |       20 |
| ADK Agent (Standard)              | no_scaling             | 0.9567 | 0.0244 |     0.9452 |     0.9681 |       20 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Standard)              | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Standard)              | no_tuning              | 0.9333 | 0.0882 |     0.7143 |     1.1524 |        3 |
| ADK Agent (Standard)              | no_tuning              | 0.9333 | 0.0882 |     0.7143 |     1.1524 |        3 |
| ADK Agent (Standard)              | no_tuning              | 0.9333 | 0.0882 |     0.7143 |     1.1524 |        3 |
| ADK Agent (Standard)              | no_tuning              | 0.9333 | 0.0882 |     0.7143 |     1.1524 |        3 |
| ADK Agent (Standard)              | full                   | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | no_ensemble            | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | full                   | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | no_ensemble            | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | full                   | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | no_ensemble            | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | full                   | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | no_ensemble            | 0.9222 | 0.1072 |     0.6560 |     1.1884 |        3 |
| ADK Agent (Standard)              | no_tuning              | 0.9117 | 0.0487 |     0.8889 |     0.9345 |       20 |
| ADK Agent (Standard)              | no_tuning              | 0.9117 | 0.0487 |     0.8889 |     0.9345 |       20 |
| ADK Agent (Standard)              | no_tuning              | 0.9117 | 0.0487 |     0.8889 |     0.9345 |       20 |
| ADK Agent (Standard)              | full                   | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | no_ensemble            | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | full                   | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | no_ensemble            | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | full                   | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | no_ensemble            | 0.9083 | 0.0494 |     0.8852 |     0.9315 |       20 |
| ADK Agent (Standard)              | baseline               | 0.3333 | 0.0000 |     0.3333 |     0.3333 |       20 |
| ADK Agent (Standard)              | baseline               | 0.3333 | 0.0000 |     0.3333 |     0.3333 |       20 |
| ADK Agent (Wrapper Test)          | full                   | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_scaling             | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_tuning              | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_ensemble            | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | minimal                | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | full                   | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_scaling             | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_tuning              | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_ensemble            | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | minimal                | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | full                   | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_scaling             | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_feature_engineering | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_tuning              | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | no_ensemble            | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| ADK Agent (Wrapper Test)          | minimal                | 0.9444 | 0.0385 |     0.8488 |     1.0401 |        3 |
| Gemini One-Shot (2.5_flash)       | no_feature_engineering | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (2.5_flash)       | minimal                | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (2.5_flash)       | no_scaling             | 0.9556 | 0.0385 |     0.8599 |     1.0512 |        3 |
| Gemini One-Shot (2.5_flash)       | full                   | 0.9111 | 0.0509 |     0.7846 |     1.0376 |        3 |
| Gemini One-Shot (2.5_flash)       | no_tuning              | 0.9111 | 0.0509 |     0.7846 |     1.0376 |        3 |
| Gemini One-Shot (2.5_flash)       | no_ensemble            | 0.9111 | 0.0509 |     0.7846 |     1.0376 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | minimal                | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | no_feature_engineering | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (2.5_flash_lite)  | no_scaling             | 0.9556 | 0.0509 |     0.8291 |     1.0820 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | full                   | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | no_tuning              | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (2.5_flash_lite)  | no_ensemble            | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (2.5_pro)         | minimal                | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (2.5_pro)         | no_scaling             | 0.9667 | 0.0333 |     0.8839 |     1.0495 |        3 |
| Gemini One-Shot (2.5_pro)         | no_feature_engineering | 0.9556 | 0.0192 |     0.9077 |     1.0034 |        3 |
| Gemini One-Shot (2.5_pro)         | full                   | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (2.5_pro)         | no_tuning              | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (2.5_pro)         | no_ensemble            | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live)            | no_scaling             | 0.9667 | 0.0333 |     0.8839 |     1.0495 |        3 |
| Gemini One-Shot (live)            | no_feature_engineering | 0.9556 | 0.0192 |     0.9077 |     1.0034 |        3 |
| Gemini One-Shot (live)            | minimal                | 0.9556 | 0.0192 |     0.9077 |     1.0034 |        3 |
| Gemini One-Shot (live)            | full                   | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live)            | no_tuning              | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live)            | no_ensemble            | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live_flash)      | no_scaling             | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (live_flash)      | minimal                | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (live_flash)      | full                   | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (live_flash)      | no_feature_engineering | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (live_flash)      | no_tuning              | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (live_flash)      | no_ensemble            | 0.9667 | 0.0000 |   nan      |   nan      |        3 |
| Gemini One-Shot (live_flash_lite) | minimal                | 0.9889 | 0.0192 |     0.9411 |     1.0367 |        3 |
| Gemini One-Shot (live_flash_lite) | no_feature_engineering | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (live_flash_lite) | no_scaling             | 0.9556 | 0.0509 |     0.8291 |     1.0820 |        3 |
| Gemini One-Shot (live_flash_lite) | full                   | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live_flash_lite) | no_tuning              | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live_flash_lite) | no_ensemble            | 0.9333 | 0.0333 |     0.8505 |     1.0161 |        3 |
| Gemini One-Shot (live_pro)        | full                   | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (live_pro)        | no_scaling             | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (live_pro)        | no_tuning              | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (live_pro)        | no_ensemble            | 0.9778 | 0.0192 |     0.9300 |     1.0256 |        3 |
| Gemini One-Shot (live_pro)        | no_feature_engineering | 0.9556 | 0.0192 |     0.9077 |     1.0034 |        3 |
| Gemini One-Shot (live_pro)        | minimal                | 0.9556 | 0.0192 |     0.9077 |     1.0034 |        3 |