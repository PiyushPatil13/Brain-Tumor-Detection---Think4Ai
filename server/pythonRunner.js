const { spawn } = require("child_process");

function runModel(imagePath) {
    return new Promise((resolve, reject) => {
        const python = spawn("python", ["model.py", imagePath]);

        let output = "";
        let error = "";

        python.stdout.on("data", (data) => {
            output += data.toString();
        });

        python.stderr.on("data", (data) => {
            error += data.toString();
        });

        python.on("close", (code) => {
            if (code !== 0) {
                return reject(error);
            }

            try {
                const result = JSON.parse(output);
                resolve(result);
            } catch {
                reject("Invalid JSON from Python");
            }
        });
    });
}

module.exports = runModel;

