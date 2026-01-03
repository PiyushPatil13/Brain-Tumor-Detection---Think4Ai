async function main() {
    // Simple fade-in on load
    window.addEventListener("load", () => {
        document.querySelectorAll(".about-section").forEach((section, index) => {
            section.style.opacity = "0";
            section.style.transform = "translateY(20px)";

            setTimeout(() => {
                section.style.transition = "0.4s ease";
                section.style.opacity = "1";
                section.style.transform = "translateY(0)";
            }, index * 100);
        });
    });
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
}
main()
