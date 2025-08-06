# 🛠 Installation & Setup Instructions

*Professional Setup Guide for Gemma Data Assistant on Windows*

This comprehensive guide will walk you through setting up the Gemma Data Assistant on your Windows system. Please follow each step carefully to ensure optimal performance and functionality.

---

## 📋 *Prerequisites & System Requirements*

### *User Responsibilities:*
Before proceeding with the installation, ensure your Windows system meets the following requirements:

- *Operating System*: Windows 10/11 (64-bit)
- *RAM*: Minimum 8GB, 16GB recommended
- *Storage*: At least 10GB free space
- *Internet*: Required for initial setup and model downloads

---

## ⚡ *Step 1: GPU Acceleration Setup (Recommended)*

### *NVIDIA GPU Users:*
If you have a modern NVIDIA GPU, installing CUDA Toolkit will significantly improve performance:

1. *Check GPU Compatibility*: Ensure you have a CUDA-capable NVIDIA GPU
2. *Download CUDA Toolkit*: Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
3. *Install CUDA*: Follow the installation wizard with default settings
4. *Verify Installation*: Open Command Prompt and run nvcc --version

> *⚠ Note*: CUDA installation is optional but highly recommended for faster AI model inference.

---

## 🐍 *Step 2: Python Installation*

### *Install Python 3.11.0 (Recommended Version):*

1. *Download Python*: Go to [python.org](https://www.python.org/downloads/release/python-3110/)
2. *Run Installer*: Download the Windows installer (64-bit)
3. *Installation Options*:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install for all users"
   - Choose "Customize installation" for advanced options
4. *Verify Installation*: Open Command Prompt and run:
   
```
python --version
pip --version
```
   

---

## 🤖 *Step 3: Ollama Installation & Model Setup*

### *Install Ollama:*
1. *Download Ollama*: Visit the official download link:
   
   https://ollama.com/download/OllamaSetup.exe
   
2. *Run Installation*: Execute the downloaded file and follow the setup wizard
3. *Complete Installation*: Restart your computer if prompted

### *Download Required AI Models:*
Open *Command Prompt* or *PowerShell* as Administrator and run these commands one by one:


ollama pull gemma3n:e2b

Wait for the first model to complete, then run:

ollama pull nomic-embed-text:v1.5


### *Verify Model Installation:*
Check if models are properly installed by running:

ollama list


*Expected Output:*

NAME                     ID              SIZE      MODIFIED
gemma3n:e2b              abc123def456    5.6 GB    2 hours ago
nomic-embed-text:v1.5    789xyz012345    274 MB    2 hours ago


### *Start Ollama Service:*
Keep Ollama running in the background:

ollama serve


> *💡 Tip*: Keep this terminal window open and running in the background for optimal performance. You can minimize it but don't close it.

---

## 📥 *Step 4: Download Gemma Data Assistant*

### *Method 1: Git Clone (Recommended)*
Open a new Command Prompt or PowerShell window and run:

git clone https://github.com/MarvelBoy047/Gemma_Ai_DataAssistant.git


### *Method 2: Direct Download*
1. Visit: [https://github.com/MarvelBoy047/Gemma_Ai_DataAssistant](https://github.com/MarvelBoy047/Gemma_Ai_DataAssistant)
2. Click *"Code"* → *"Download ZIP"*
3. Extract the ZIP file to your desired location

---

## 🔧 *Step 5: Install Dependencies*

Navigate to the project directory and install required packages:

cd Gemma_Ai_DataAssistant
pip install -r requirements.txt


> *⏳ Note*: This process may take 5-10 minutes depending on your internet speed.

---

## 🚀 *Step 6: Launch the Application*

### *Start the Application:*
In the project directory, run:

streamlit run app.py


### *Initial Setup Process:*
![Initial Setup](assets/1.png)

*First-Time Launch:*
- The application will index the knowledge base (one-time process)
- This initial setup takes 2-3 minutes
- Subsequent launches will be much faster (10-15 seconds)

### *Access the Application:*
- Your default browser will automatically open
- Navigate to: http://localhost:8501
- The application interface will be ready for use

---

## 📺 *Step 7: Usage Instructions & Tutorial*

For comprehensive usage instructions and tutorials, watch our detailed video guide:

[![Gemma Data Assistant Tutorial](https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

*Watch the complete tutorial:* [Gemma Data Assistant - Complete Setup & Usage Guide](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

*Video Preview:*

*Alternative*: Explore the intuitive interface on your own - it's designed to be user-friendly and self-explanatory.

---

## ✅ *System Compatibility & Testing*

### *Tested Configurations:*
- *✅ NVIDIA RTX 2060*: Excellent performance, fast response times
- *✅ AMD CPU-only*: Fair performance, longer processing times but fully functional
- *✅ Windows 10/11*: Fully supported and tested

### *Compatibility Notes:*
- *❓ Other Operating Systems*: Not tested on macOS or Linux
- *⚠ Use Caution*: If using on untested systems, monitor performance carefully

---

## ⚠ *Known Limitations & Troubleshooting*

### *Disk Space Management:*
If you experience disk space issues after extended use:

1. *Clear Temporary Files*:
   - Press Windows + R
   - Type %temp% and press Enter
   - Delete as many files as possible
   - Some files may be locked (this is normal)

2. *Clear Application Cache*:
   - Navigate to the project folder
   - Delete contents of agent_memory/ and chat_history/ if needed

### *Performance Considerations:*
- *Task Size Impact*: Smaller analysis requests = faster responses
- *Memory Usage*: Large datasets may require more RAM
- *Concurrent Sessions*: One analysis per chat session (future updates will address this)

### *Common Issues:*
- *Slow Response*: Ensure CUDA is properly installed for GPU acceleration
- *Memory Errors*: Close other applications to free up RAM
- *Port Conflicts*: If port 8501 is busy, Streamlit will suggest an alternative
- *Model Loading Issues*: Restart Ollama service if models fail to load

---

## 🔄 *Future Updates & Patches*

### *Planned Improvements:*
- *Multiple Analyses*: Support for multiple notebooks per chat session
- *Cross-Platform*: macOS and Linux compatibility
- *Performance*: Enhanced memory management and speed optimizations
- *Features*: Additional analysis templates and visualization options

### *Update Process:*
- Regular patches will be released via GitHub
- Run git pull in the project directory to get latest updates
- Check our repository for release notes and changelog

---

## 📞 *Support & Assistance*

### *If You Encounter Issues:*
1. *Check System Requirements*: Ensure all prerequisites are met
2. *Restart Services*: Try restarting Ollama and the application
3. *Review Logs*: Check terminal output for error messages
4. *GitHub Issues*: Report bugs at our repository
5. *Community Support*: Join our Discord server for help

### *Performance Tips:*
- *Close Unnecessary Applications*: Free up system resources
- *Use SSD Storage*: Faster disk access improves performance
- *Monitor Temperature*: Ensure your system doesn't overheat during intensive tasks
- *Regular Updates*: Keep Python and dependencies updated

---

*🎉 Congratulations! You're now ready to harness the power of AI-driven data analysis with complete privacy and professional-grade results.*
```
