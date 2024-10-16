const vscode = require('vscode');
const axios = require('axios');

async function generateLogAdviceLLM() {
	const editor = vscode.window.activeTextEditor;
	if (!editor) return;

	const selectedText = editor.document.getText(editor.selection);
	if (!selectedText) {
		vscode.window.showInformationMessage("Please select some code.");
		return;
	}

	const prompt = (
		"You are an AI assistant that helps developers improve their logging practices. "
		+ "Based on the following code snippet, improve with better logging practices: "
		+ "Code Snippet:\n\n"
		+ selectedText
	);

	try {
		console.log("Calling the LLM model to get code suggestion with the selected text: ", selectedText);
		const response = await axios.post('http://localhost:8080/completion', {
			prompt: prompt,
			max_tokens: 100,
			temperature: 0.1,
		}, {
			headers: {
				'Content-Type': 'application/json',
			}
		});
		console.log("Response from LLM model: ", response.data);

		const suggestedCode = response.data.content;

		// Open a webview panel to show the generated code and buttons
		openWebviewWithCodeSuggestion(suggestedCode, editor);

	} catch (error) {
		console.error(error);
		vscode.window.showErrorMessage("Failed to get code suggestion.");
	}
}

function openWebviewWithCodeSuggestion(suggestedCode, editor) {
	// Create a new webview panel
	const panel = vscode.window.createWebviewPanel(
		'logAdvice', // Identifies the type of the webview
		'Log Advice Suggestion', // Title displayed to the user
		vscode.ViewColumn.One, // Editor column to show the new webview panel in
		{
			enableScripts: true // Allow scripts in the webview
		}
	);

	// HTML content of the webview
	panel.webview.html = getWebviewContent(suggestedCode);

	// Handle messages from the webview (for buttons)
	panel.webview.onDidReceiveMessage(async message => {
		switch (message.command) {
			case 'accept':
				// Insert the suggested code in the editor
				editor.edit(editBuilder => {
					editBuilder.replace(editor.selection, suggestedCode);
				});
				panel.dispose(); // Close the webview panel
				break;
			case 'decline':
				panel.dispose(); // Close the webview panel without doing anything
				break;
		}
	});
}

// Function to generate HTML content for the webview
function getWebviewContent(suggestedCode) {
	return `
		<!DOCTYPE html>
		<html lang="en">
		<head>
			<meta charset="UTF-8">
			<meta name="viewport" content="width=device-width, initial-scale=1.0">
			<title>Code Suggestion</title>
			<style>
				body {
					font-family: Arial, sans-serif;
					padding: 20px;
				}
				#code {
					width: 100%;
					height: 200px;
					background-color: #f5f5f5;
					border: 1px solid #ccc;
					overflow: auto;
					white-space: pre;
					padding: 10px;
				}
				.button-container {
					margin-top: 20px;
				}
				button {
					margin-right: 10px;
					padding: 10px 15px;
					font-size: 14px;
					cursor: pointer;
				}
			</style>
		</head>
		<body>
			<h2>Suggested Code Improvement</h2>
			<div id="code">${suggestedCode}</div>
			<div class="button-container">
				<button id="accept">Accept</button>
				<button id="decline">Decline</button>
			</div>

			<script>
				// Handle button clicks and send messages to the extension
				const vscode = acquireVsCodeApi();
				document.getElementById('accept').addEventListener('click', () => {
					vscode.postMessage({ command: 'accept' });
				});
				document.getElementById('decline').addEventListener('click', () => {
					vscode.postMessage({ command: 'decline' });
				});
			</script>
		</body>
		</html>
	`;
}

// This method is called when your extension is activated
function activate(context) {

	context.subscriptions.push(vscode.commands.registerCommand('log-advice-generator.generateLogAdvice', generateLogAdviceLLM));

	context.subscriptions.push(vscode.commands.registerCommand('log-advice-generator.helloWorld', function () {
		vscode.window.showInformationMessage('Hello World from Log Advice Generator!');
	}));
}

function deactivate() { }

module.exports = {
	activate,
	deactivate
};
