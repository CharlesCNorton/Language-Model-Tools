from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
import traceback
import tkinter as tk
from tkinter import filedialog
import re

inferno_lm = None

class StopOnUser(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_token_id = self.tokenizer.encode("User:", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids[0][-len(self.stop_token_id):].tolist() == self.stop_token_id:
            return True
        return False

class InfernoLM:
    def __init__(self, device="cuda", precision="float16", model_path=None, verbose=False):
        self.device = device
        self.precision = precision
        self.model_path = model_path
        if model_path:
            self.tokenizer, self.model = self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device=self.device, dtype=torch.float16 if self.precision == "float16" else torch.float32)
        return tokenizer, model

    def generate_response(self, messages, max_length=1000):
        try:
            user_input = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
            context_history = ""
            for message in messages:
                if message["role"] == "user":
                    context_history += f"User: {message['content']}\n"
                elif message["role"] == "assistant":
                    context_history += f"Assistant: {message['content']}\n"

            inputs = self.tokenizer.encode_plus(context_history, return_tensors='pt', padding=True, truncation=True, max_length=4096)
            inputs = inputs.to(self.device)

            stop_on_user = StopOnUser(self.tokenizer)

            generation_config = {
                'input_ids': inputs['input_ids'],
                'max_length': len(inputs['input_ids'][0]) + max_length,
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'pad_token_id': self.tokenizer.eos_token_id,
                'do_sample': True,
                'num_beams': 3,
                'early_stopping': True,
                'stopping_criteria': StoppingCriteriaList([stop_on_user])
            }

            outputs = self.model.generate(**generation_config)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return self.extract_response(generated_text, user_input)
        except Exception as e:
            error_message = f"An error occurred during generation: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return error_message

    def extract_response(self, generated_text, user_input):
        try:
            print(f"Attempting to extract response from: {generated_text}")

            split_text = generated_text.split(f"User: {user_input}")
            if len(split_text) > 1:
                extracted = split_text[1].split("User:")[0].strip()
                extracted = re.sub(r'^Assistant:\s*', '', extracted).strip()
                print(f"Extracted response: {extracted}")
                return extracted
            else:
                error_msg = "I'm sorry, I couldn't generate a proper response."
                print(error_msg)
                return error_msg
        except Exception as e:
            error_message = f"An error occurred during response extraction: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return error_message

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chat with InfernoLM</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            body {
                background-color: #121212;
                color: #E0E0E0;
                font-family: 'Roboto', sans-serif;
            }
            #chat-container {
                width: 80%;
                margin: 30px auto;
                padding: 20px;
                background: #1E1E1E;
                border-radius: 8px;
            }
            #messages {
                height: 600px;
                overflow-y: auto;
                border: 1px solid #333333;
                padding: 10px;
                background: #262626;
                border-radius: 5px;
                line-height: 1.6;
            }
            .message {
                padding: 10px;
                margin-bottom: 15px;
                line-height: 1.6;
            }
            .user {
                background-color: #4A90E2;
                align-self: flex-end;
            }
            .bot {
                background-color: #333333;
                align-self: flex-start;
            }
            #user-input, input[type="number"], button {
                width: auto;
                padding: 10px;
                margin-top: 20px;
                background: #333333;
                border: 1px solid #474747;
                color: #E0E0E0;
                border-radius: 5px;
            }
            button {
                padding: 10px;
                margin-top: 10px;
                background-color: #4A4A4A;
                color: #E0E0E0;
                border: none;
                border-radius: 5px;
                cursor: pointer.
            }
            pre, code {
                font-size: 0.9em;
                background-color: #262626;
                color: #E0E0E0;
                padding: 10px;
                border-radius: 5px;
                white-space: pre-wrap;
                line-height: 1.6;
                font-family: 'Consolas', 'Monaco', 'Courier New', Courier, monospace;
                overflow-x: auto.
            }
            code {
                font-family: 'Consolas', 'Monaco', 'Courier New', Courier, monospace.
            }
            .loader {
                border: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                display: none;
                position: absolute;
                left: 50%;
                top: 50%;
                transform: translate(-50%, -50%);
            }
        </style>
    </head>
    <body>
        <div id="chat-container">
            <div id="messages"></div>
            <input type="text" id="user-input" placeholder="Type your message here...">
            <button onclick="sendMessage()">Send Message</button>
            <div class="loader" id="loading-indicator"></div>
        </div>
        <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
        <script>
            var conversationHistory = [];
            $(document).ready(function() {
                $("#user-input").keypress(function(e) {
                    if(e.which == 13) {
                        sendMessage();
                    }
                });
            });

            function sendMessage() {
                var inputText = $('#user-input').val();
                if(inputText) {
                    $('#loading-indicator').show();
                    conversationHistory.push({role: 'user', content: inputText});
                    $('#user-input').val('');
                    $("#messages").append('<div class="message user">' + inputText + '</div>');
                    $.ajax({
                        url: '/chat',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ messages: conversationHistory }),
                        success: function(data) {
                            $('#loading-indicator').hide();
                            $("#messages").append('<div class="message bot">' + data.content + '</div>');
                            conversationHistory.push({role: 'assistant', content: data.content});
                            $("#messages").scrollTop($("#messages")[0].scrollHeight);
                        },
                        error: function(response) {
                            $('#loading-indicator').hide();
                            $("#messages").append('<div class="message bot">Error: ' + response.responseText + '</div>');
                        }
                    });
                }
            }
        </script>
    </body>
    </html>
    """)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        messages = data["messages"]
        response_content = inferno_lm.generate_response(messages)
        if isinstance(response_content, str):
            return jsonify({"content": "Assistant: " + response_content})
        else:
            return jsonify({"error": "Failed to generate a response"}), 500
    except Exception as e:
        error_message = f"An error occurred in /chat route: {str(e)}"
        print(error_message)
        traceback.print_exc()
        return jsonify({"error": error_message}), 500

@app.route('/back_message', methods=['POST'])
def back_message():
    return jsonify({"success": False, "message": "No messages to remove"}), 400

@app.route('/clear_history', methods=['POST'])
def clear_history():
    return jsonify({"success": True, "message": "History cleared"})

def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()
    return folder_selected

if __name__ == '__main__':
    model_path = select_folder()
    inferno_lm = InfernoLM(model_path=model_path, device="cuda", precision="float16")

    choice = input("Run server for local access only (1) or network-wide access (2)? Enter 1 or 2: ")
    if choice == '2':
        print("Server will be accessible network-wide.")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    else:
        print("Server will only be accessible locally.")
        app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
