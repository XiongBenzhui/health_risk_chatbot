### **Hello, my friend!**
# **Health Risk Assessment Chatbot🤖**

Welcome to the Health Risk Chatbot project! The large language model chatbot helps you assess health risks and receive personalized health protection advice in the daily activities. 
Follow the instructions below to set up and run the chatbot.

## **Prerequisites**

**Python 3.12**: The programming language used for the project.

## **Install**

### 1. **Create a New Conda Environment**:
Open your terminal and execute the following command to create a new environment named `healthRisk` with Python 3.12:
```bash
conda create -n healthRisk python=3.12
```
### 2. **Activate the Environment**:
Activate the newly created environment:
```bash
conda activate healthRisk
```
### 3. **Clone the Health Risk Chatbot Repository**:
Clone the repository containing the chatbot code:
```bash
git clone https://github.com/XiongBenzhui/health_risk_chatbot.git
cd health_risk_chatbot
```
### 4. **Download Open-source Models**:
#### Multilingual E5 Large Instruct Model:
This model is hosted on Hugging Face. To download it:
```bash
git clone https://huggingface.co/intfloat/multilingual-e5-large-instruct.git
```
Or find the model from https://huggingface.co/intfloat/multilingual-e5-large-instruct and download it file by file. After downloading, removing the model into ‘models’ file.
#### DeepSeek-R1:32B Model:
This model is available through Ollama. To download and run it:
```bash
ollama pull deepseek-r1:32b
```
Ensure that Ollama is installed and properly configured. For more information, visit the Ollama website. You can change to any chat model you want.
### 5. **Install PyTorch with CUDA Support**:
Install the appropriate versions of PyTorch, torchvision, and torchaudio depending on your server.
Our command is:
```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```
### 6. **Install Additional Dependencies**:
Install the remaining required Python packages:
```bash
pip install -r requirements.txt
```

## **Usage**

### 7. **Run the Chatbot**:
Start the chatbot by executing:
```bash
python health_risk_chatbot.py
```
#### Access the Chatbot in Your Browser:
Open your web browser and navigate to the web. Here, you can:

- Ask LLM: Interact with the model without knowledge enhancement.

- Ask LLM-RAG: Use the model with knowledge enhancement for more informed responses.

Please be patient while the assistant processes your request. Avoid submitting multiple queries or refreshing the page during this time to ensure optimal performance.

## **Contributing**
We welcome contributions to improve the Health Risk Chatbot. If you have suggestions or find issues, please open an issue or submit a pull request.

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.

## **Acknowledgments**
PyTorch team for developing the PyTorch library.

All contributors who have helped improve this project.

## **Thank you for using the Health Risk Chatbot! ♥**
