const vscode = require('vscode');
const axios = require('axios');

let MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

async function callBackendPost(path, args) {
	const URL = 'http://localhost:8888'
	return await axios.post(URL + path, args, {
		headers: {
			'Content-Type': 'application/json',
		}
	});
}

async function callBackendGet(path, params) {
	const URL = 'http://localhost:8888'
	let parameters = ""
	if (params != null) {
		parameters = '?' + params
	}
	return await axios.get(URL + path + parameters);
}

async function generateLogAdvice() {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showInformationMessage("No active editor found.");
        return;
    }

    const selectedText = editor.document.getText(editor.selection);
    if (!selectedText) {
        vscode.window.showInformationMessage("Please select some code.");
        return;
    }

    const prompt = (
        "Context: You are an AI assistant that helps people with their questions. "
        + "Please only add 2 to 5 lines of code to improve log messages to the following code: "
        + selectedText
    );

    // Show loading progress window while waiting for the response
    vscode.window.withProgress({
        location: vscode.ProgressLocation.Notification,
        title: "Generating Log Advice",
        cancellable: false
    }, async (progress) => {
        progress.report({ message: "Contacting LLM..." });

        try {
            console.log("Calling the LLM model to get code suggestion with the selected text: ", selectedText);
            
			// Call your LLM service
			const response = await callBackendPost('/predict', {prompt: prompt, max_new_tokens: 100, temperature: 0.1}) 
            

            console.log("Response from LLM model: ");
			console.log(JSON.stringify(response.data, null, 2))
            const suggestedCode = response.data.content;

            // Create a text edit with the generated code
            const edit = new vscode.WorkspaceEdit();
            const range = new vscode.Range(editor.selection.start, editor.selection.end);
            edit.replace(editor.document.uri, range, suggestedCode);

            // Apply the edit as a preview
            await vscode.workspace.applyEdit(edit);

            // Prompt the user to accept or decline the changes
            const userResponse = await vscode.window.showInformationMessage(
                "Log advice generated. Do you want to apply the changes?",
                "Yes",
                "No"
            );

            if (userResponse === "Yes") {
                // Apply the changes permanently
                await editor.edit(editBuilder => {
                    editBuilder.replace(editor.selection, suggestedCode);
                });
                vscode.window.showInformationMessage("Log advice applied.");
            } else {
                // Revert the changes
                vscode.commands.executeCommand('undo');
                vscode.window.showInformationMessage("Log advice discarded.");
            }
        } catch (error) {
            console.error(error);
            vscode.window.showErrorMessage("Failed to get code suggestion.");
        }
    });
}

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed

/**
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    let disposable = vscode.commands.registerCommand('log-advice-generator.generateLogAdvice', generateLogAdvice);
    context.subscriptions.push(disposable);

	let changeModel = vscode.commands.registerCommand('log-advice-generator.changeModelId', async () => {
        const model = await vscode.window.showInputBox({ prompt: 'Enter a HuggingFace Model ID', value: MODEL_ID });
        if (model) {
            MODEL_ID = model;
            console.log(`Changing Model to : ${MODEL_ID}`)
            const response = await callBackendPost('/change_model', {model_id: model})
			console.log(JSON.stringify(response.data, null, 2))
            if (response.data.completed == true) {
                vscode.window.showInformationMessage('Model Change has been successfull, Model configured : ' + response.data.model_name)
            } else {
				vscode.window.showErrorMessage('Model Change Failed, Model configured : ' + response.data.model_name);
			}
        } else {
            vscode.window.showErrorMessage('MODEL ID is required');
        }


    });
	context.subscriptions.push(changeModel);

	let changeToken = vscode.commands.registerCommand('log-advice-generator.changeToken', async () => {
        const token = await vscode.window.showInputBox({ prompt: 'Enter a HuggingFace Model ID'});
        if (token) {
            console.log(`Changing token`)
            const response = await callBackendPost('/change_token', {token: token})
			console.log(JSON.stringify(response.data, null, 2))
            if (response.data.completed == true) {
                vscode.window.showInformationMessage('Token Change has been successfull')
            } else {
				vscode.window.showErrorMessage('Token Change Failed');
			}
        } else {
            vscode.window.showErrorMessage('TOKEN is required');
        }
    });
	context.subscriptions.push(changeToken);

	let getModelInfo = vscode.commands.registerCommand('log-advice-generator.modelInfo', async () => {
		const response = await callBackendGet('/model_info', null)
		console.log(JSON.stringify(response.data, null, 2))
        vscode.window.showInformationMessage('The model configured is [' + response.data.model_name + ']')
		vscode.window.showInformationMessage('Running on [' + response.data.device + ']')


    });
	context.subscriptions.push(getModelInfo);

	
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};