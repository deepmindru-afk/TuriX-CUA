<p align="center">
   <img src="./doc/LogoHFitted.svg" width="1600" alt="TuriX logo">
</p>

<h1 align="center">TuriX · Desktop Actions, Driven by AI</h1>

<p align="center"><strong>Talk to your computer, watch it work.</strong></p>

## 📞 Contact & Community

Join our Discord community for support, discussions, and updates:

<p align="center">
   <a href="https://discord.gg/BRxXtnGg">
      <img src="https://img.shields.io/discord/1400749393841492020?color=7289da&label=Join%20our%20Discord&logo=discord&logoColor=white&style=for-the-badge" alt="Join our Discord">
   </a>
</p>

Or contact us with email: contact@turix.ai

---

TuriX lets your powerful AI models take real, hands‑on actions directly on your desktop. 
It ships with a **state‑of‑the‑art computer‑use agent** (passes > 68 % of our internal OSWorld‑style test set) yet stays 100 % open‑source and cost‑free for personal & research use.  

Prefer your own model? **Change in `config.json` and go.**

## Table of Contents
- [📞 Contact & Community](#-contact--community)
- [📰 Latest News](#-latest-news)
- [🖼️ Demos](#️-demos)
- [✨ Key Features](#-key-features)
- [📊 Model Performance](#-model-performance)
- [🚀 Quick‑Start (Windows)](#-quickstart-windows)
   - [1. Download the App](#1-download-the-app)
   - [2. Create a Python 3.12 Environment](#2-create-a-python-312-environment)
   - [3. Configure & Run](#3-configure--run)
- [🤝 Contributing](#-contributing)
- [🗺️ Roadmap](#️-roadmap)

---

## 📰 Latest News

**December 30, 2025** - 🎉 Significant update in Agent Architecture. We introduce a multi-model architecture in the multi-agent branch, releasing the stress from a single model to multiple models.

**October 16, 2025** - 🚀 Big news for automation enthusiasts! TuriX now fully supports the cutting-edge **Qwen3-VL** vision-language model, empowering seamless PC automation across both **macOS** and **Windows**. This integration boosts task success rates by up to 15% on complex UI interactions (based on our internal benchmarks), making your desktop workflows smarter and faster than ever. Whether you're scripting daily routines or tackling intricate projects, Qwen3-VL's advanced multimodal reasoning brings unparalleled precision to the table.
Ready to level up? Update your `config.json` and start automating—happy hacking! 🎉

**September 30, 2025** - 🎉 Exciting update! We've just released our latest AI model on the [TuriX API platform](https://turixapi.io), bringing enhanced performance, smarter reasoning, and seamless integration for even more powerful desktop automation. Developers and researchers, this is your cue—head over to the platform to access it now and elevate your workflows!

*Stay tuned to our [Discord](https://discord.gg/vkEYj4EV2n) for tips, user stories, and the next big drop.*

---

## 🖼️ Demos
<h3 align="center">MacOS Demo</h3>
<p align="center"><strong>Book a flight, hotel and uber.</strong></p>
<p align="center">
   <img src="./doc/booking_demo.gif" width="1600" alt="TuriX macOS demo - booking">
</p>

<p align="center"><strong>Search iPhone price, create Pages document, and send to contact</strong></p>
<p align="center">
   <img src="./doc/demo1.gif" width="1600" alt="TuriX macOS demo - iPhone price search and document sharing">
</p>

<p align="center"><strong>Generate a bar-chart in the numbers file sent by boss in discord and insert it to the right place of my powerpoint, and reply my boss.</strong></p>
<p align="center">
   <img src="./doc/complex_demo_mac.gif" width="1600" alt="TuriX macOS demo - excel graph to powerpoint">
</p>

<h3 align="center">Windows Demo</h3>
<p align="center"><strong>Search video content in youtube and like it</strong></p>
<p align="center">
   <img src="./doc/win_demo1.gif" width="1600" alt="TuriX Windows demo - video search and sharing">
</p>

<h3 align="center">MCP with Claude Demo</h3>
<p align="center"><strong>Claude search for AI news, and call TuriX with MCP, write down the research result to a pages document and send it to contact</strong></p>
<p align="center">
   <img src="./doc/mcp_demo1.gif" width="1600" alt="TuriX MCP demo - news search and sharing">
</p>

---

## ✨ Key Features
| Capability | What it means |
|------------|---------------|
| **SOTA default model** | Outperforms previous open‑source agents (e.g. UI‑TARS) on success rate and speed on Mac |
| **No app‑specific APIs** | If a human can click it, TuriX can too—WhatsApp, Excel, Outlook, in‑house tools… |
| **Hot‑swappable "brains"** | Replace the VLM policy without touching code (`config.json`) |
| **MCP‑ready** | Hook up *Claude for Desktop* or **any** agent via the Model Context Protocol (MCP) |

---
## 📊 Model Performance

Our agent achieves state-of-the-art performance on desktop automation tasks:
<p align="center">
   <img src="./doc/performance_sum.jpg" width="1600" alt="TuriX performance">
</p>

For more details, check our [report](https://turix.ai/technical-report/).

## 🚀 Quick‑Start (Windows)

> **We never collect data**—install, grant permissions, and hack away.

> **0. Mac Users**: Switch to the `multi-agent` branch for MacOS-specific setup and installation instructions.
>
> ```bash
> git checkout multi-agent
> ```

### 1. Download the App
For easier MacOS usage, [download the app](https://turix.ai/)
The app only works for Mac!!!

Or follow the manual setup below:

### 2. Create a Python 3.12 Environment
Firstly Clone the repository and run:
```bash
conda create -n turix_env python=3.12
conda activate turix_env        # requires conda ≥ 22.9
pip install -r requirements.txt
```

> **Click "Allow" on every dialog** so the agent can run.

### 3. Configure & Run

#### 3.1 Edit Task Configuration

Edit task in `examples/config.json`:

> [!IMPORTANT]
> **Task Configuration is Critical**: The quality of your task instructions directly impacts success rate. Clear, specific prompts lead to better automation results.

```json
{
    "agent": {
         "task": "open Chrome, go to github, search for TuriX CUA, enter the TuriX repository, and star this repository. "
    }
}
```
There is no use_ui parameter in the windows version, the state is only a screenshot.

#### 3.2 Edit API Configuration

Get API now with $20 credit from our [official web page](https://turix.ai/api-platform/).
Login to our website and the key is at the bottom.

Edit API in `examples/config.json`:
```json
"brain_llm": {
      "provider": "turix",
      "model_name": "turix-brain-model",
      "api_key": "YOUR_API_KEY",
      "base_url": "https://llm.turixapi.io/v1"
   },
"actor_llm": {
      "provider": "turix",
      "model_name": "turix-actor-model",
      "api_key": "YOUR_API_KEY",
      "base_url": "https://llm.turixapi.io/v1"
   },
"memory_llm": {
      "provider": "turix",
      "model_name": "turix-memory-model",
      "api_key": "YOUR_API_KEY",
      "base_url": "https://llm.turixapi.io/v1"
   },
"planner_llm": {
      "provider": "turix",
      "model_name": "turix-planner-model",
      "api_key": "YOUR_API_KEY",
      "base_url": "https://llm.turixapi.io/v1"
   }
```

In this multi-agent branch, you need to set the brain, actor, and memory models. If you enable planning
(`agent.use_plan: true`), you also need to set the planner model.
We strongly recommend you set the turix-actor model as the actor. The brain can be any VLM you like.

For a local Ollama setup, point each role to your Ollama server:
```json
"brain_llm": {
      "provider": "ollama",
      "model_name": "llama3.2-vision",
      "base_url": "http://localhost:11434"
   },
"actor_llm": {
      "provider": "ollama",
      "model_name": "llama3.2-vision",
      "base_url": "http://localhost:11434"
   },
"memory_llm": {
      "provider": "ollama",
      "model_name": "llama3.2-vision",
      "base_url": "http://localhost:11434"
   },
"planner_llm": {
      "provider": "ollama",
      "model_name": "llama3.2-vision",
      "base_url": "http://localhost:11434"
   }
```

#### 3.3 Configure Custom Models (Optional)

If you want to use other models not defined by the build_llm function in the main.py, you need to first define it, then setup the config.

main.py:

```
if provider == "name_you_want":
        return ChatOpenAI(
            model="gpt-4.1-mini", api_key=api_key, temperature=0.3
        )
```
Switch between ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic, or ChatOllama base on your llm. Also change the model name.

#### 3.4 Start the Agent

```bash
python examples/main.py
```

**Enjoy hands‑free computing 🎉**

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

Quick links:
- [Development Setup](CONTRIBUTING.md#development-setup)
- [Code Style Guidelines](CONTRIBUTING.md#code-style-guidelines)
- [Testing](CONTRIBUTING.md#testing)
- [Pull Request Process](CONTRIBUTING.md#pull-request-process)

For bug reports and feature requests, please [open an issue](https://github.com/TurixAI/TuriX-CUA/issues).

## 🗺️ Roadmap

| Quarter | Feature | Description |
|---------|---------|-------------|
| **2025 Q3** | **✅ Windows Support** | Cross-platform compatibility bringing TuriX automation to Windows environments *(Now Available)* |
| **2025 Q3** | **✅ Enhanced MCP Integration** | Deeper Model Context Protocol support for seamless third-party agent connectivity *(Now Available)*|
| **2025 Q4** | **✅ Next-Gen AI Model** | Significantly improved reasoning and task execution capabilities |
| **2025 Q4** | **✅ Support Gemini-3-pro model** | Run with any compatible vision language models |
| **2025 Q4** | **✅ Planner** | Understands user intent and makes step-by-step plans to complete tasks |
| **2025 Q4** | **✅ Multi-Agent Architecture** | Evaluate and guide each step in working |
| **2025 Q4** | **✅ Duckduckgo Integration** | Speed up the information gathering process, for smarter planning (multi-agent branch) |
| **2026 Q1** | **✅ Ollama Support** | Support the Ollama Qwen3vl models |
| **2026 Q1** | **Recoverable Memory Compression** | Advance memory management mechanism to stabelize performance |
| **2026 Q1** | **Offline Model Option** | Fully local TuriX model for maximum privacy and zero cloud API dependency |
| **2026 Q1** | **Workflow Automation** | Record, edit, and replay complex multi-step automation sequences |
| **2026 Q2** | **Persistent Memory** | Learn user preferences and maintain task history across sessions |
| **2026 Q2** | **Learning by Demonstration** | Train the agent by showing it your preferred methods and workflows |
| **2026 Q2** | **Windows-Optimized Model** | Native Windows model architecture for superior performance on Microsoft platforms |
