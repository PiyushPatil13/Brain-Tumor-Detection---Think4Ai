    async function main() {
        
        // hamburger to cross transition
        const hamburger = document.getElementById("hamburger");
        const menu = document.querySelector(".buttons");
        const cross = document.querySelector(".cross")
        
        hamburger.addEventListener("click", () => {
            menu.classList.toggle("open");
            cross.style.display = "block"
        });
        cross.addEventListener("click",()=>{
            menu.classList.remove("open");
            cross.style.display = "none"
        })
    }
    main()
