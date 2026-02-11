const express = require('express')
const app = express()
const path = require('path')
const port = 3000
const multer = require('multer')


const fs = require("fs")
const runModel = require("./pythonRunner")
const Image = require("./models/Image")
const connectDB = require("./db")
const mongoose = require("mongoose")
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const uniqueName = Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  }
});
const upload = multer({
  storage: storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
  fileFilter: function (req, file, cb) {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only images are allowed"));
    }
  }
});

connectDB()

app.use(express.static(path.join(__dirname,'..','frontend')));

app.post("/detection", upload.single("image"), async (req, res) => {
    try {
      if (!req.file) {
            return res.status(400).json({ error: "No image uploaded" });
        }
        const imagePath = req.file.path
        
        const result = await runModel(imagePath)
        try{
            const gradcam_path = "../CNN AND GRADCAM/"
        }
        catch(err){
            console.log(err)
        }
        const doc = await Image.create({
            originalImage: imagePath,
            gradcamImage: gradcam_path,
            prediction: result.predicted_class,
            confidence: result.confidence
        })
        
        // delete the image
        fs.unlink(imagePath, () => {
            console.log("Temp image deleted")
        })

        res.json({
            prediction: result.predicted_class,
            confidence: result.confidence,
            gradcam: gradcam_path
        })
} catch (err) {
        res.status(500).json({ error: err.message })
    }
    
})

app.get('/',(req,res)=>{
    res.sendFile('../frontend/index.html')
})
app.get('/detection',async (req,res)=>{
     try {
        const results = await Image.find().sort({ createdAt: -1 })
        res.json(results)
    } catch (err) {
        res.status(500).json({ error: err.message })
    }
})


app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
