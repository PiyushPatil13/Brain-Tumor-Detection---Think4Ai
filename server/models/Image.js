const mongoose = require("mongoose")

const imageSchema = new mongoose.Schema({
    originalImage: {
        type: String,
        required: true
    },
    gradcamImage: {
        type: String
    },
    prediction: {
        type: String,
        required: true
    },
    confidence: {
        type: Number,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    }
})

module.exports = mongoose.model("Image", imageSchema)
