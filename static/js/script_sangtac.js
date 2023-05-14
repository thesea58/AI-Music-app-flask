
const selectButtons = document.querySelectorAll(".select-button");

selectButtons.forEach(button => {
    button.addEventListener('click', () => {
        const filename = button.dataset.filename;
        console.log(filename);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/process_select');
        xhr.setRequestHeader('Content-Type', 'application/json;charset=UTF-8');
        xhr.send(filename);
    });
});



const startSlider = document.getElementById('start-slider');
const endSlider = document.getElementById('end-slider');
const startValue = document.getElementById('slider-value-1');
const endValue = document.getElementById('slider-value-2');

startSlider.addEventListener('input', () => {
    startValue.textContent = startSlider.value;
});

endSlider.addEventListener('input', () => {
    endValue.textContent = endSlider.value;
});

// gửi 2 giá trị cho flask
const submitButton = document.querySelector("#sangtac");


submitButton.addEventListener("click", () => {
    
    var startValue_text = document.querySelector("#slider-value-1").textContent;
    var endValue_text = document.querySelector("#slider-value-2").textContent;
    // var inputGenLen_text = document.querySelector('#inputGenLen').val().toString();
    // ,'inputGenLen': inputGenLen_text
    const data = { 'startValue': startValue_text, 'endValue': endValue_text };
    console.log(data)
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/show_result");
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
            console.log(this.responseText);
        }
    };
    xhr.send(JSON.stringify(data));
});


