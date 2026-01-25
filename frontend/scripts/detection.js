const fileInput = document.getElementById("fileInput")
const button = document.getElementById("upload")
const baseUrl = "http://localhost:3000/"
async function sendImg(formData){
    let x = await fetch(baseUrl + "detection",{
        method: "POST",
        body: formData
    })
    const data =  await x.json()
    console.log(data);    
}
async function main() {

    const hamburger = document.getElementById("hamburger");
    const menu = document.querySelector(".buttons");
    const cross = document.querySelector(".cross")

    hamburger.addEventListener("click", () => {
        menu.classList.toggle("open");
        cross.style.display = "block"
    });
    cross.addEventListener("click", () => {
        menu.classList.remove("open");
        cross.style.display = "none"
    })
    button.addEventListener("click",async()=>{
        const file =  fileInput.files[0]
        if (!file) {
        alert("Please select an image first");
        return;
    }
        let formData =  new FormData()
        formData.append("image",file)
        try{
            await sendImg(formData)
        }
        catch(err){
            console.log("Error:",err)
        }
    })
}
main()