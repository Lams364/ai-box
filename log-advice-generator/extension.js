// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
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
		"Context: You are an AI assistant that helps people with their questions. "
		+ "Answer only the question you are being asked. Don't add questions that is not in the prompt. Be consise. "
		+ "Don't add an introduction or any form of 'A:' to your answer. Just answer the question after the 'QUESTION:' tag. "
		+ "QUESTION:\n\n"
		+ selectedText
	)

	try {
		console.log("Calling the LLM model to get code suggestion with the selected text: ", selectedText);
		const response = await axios.post('http://localhost:8888/predict', {
			text: prompt,
			max_tokens: 100,
			temperature: 0.1,
		}, {
			headers: {
				'Content-Type': 'application/json',
			}
		});
		console.log("Response from LLM model: ", response.data);

		const suggestedCode = response.data.content;

		// Insert the suggested code into the editor
		editor.edit(editBuilder => {
			editBuilder.insert(editor.selection.end, `\n\n${suggestedCode}`);
		});

	} catch (error) {
		console.error(error);
		vscode.window.showErrorMessage("Failed to get code suggestion.");
	}
}




// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
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
}
