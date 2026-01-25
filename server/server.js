const express = require('express')
const app = express()
const path = require('path')
const port = 3000
const multer = require('multer')
const upload = multer({dest:"../CNN AND GRADCAM/uploads/"})
app.use(express.static(path.join(__dirname,'..','frontend')));
app.get('/',(req,res)=>{
    res.sendFile('../frontend/index.html')
})
app.post('/detection',upload.any(), (req, res) => {
    console.log(req.file)
  res.send('Hello World!')
})

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})
