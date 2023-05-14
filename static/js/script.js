

// Hiệu ứng khi di chuyển chuột trên nền của hero section
const hero = document.querySelector('.hero');
const heroText = document.querySelector('.hero-text');

hero.addEventListener('mousemove', (e) => {
    let xAxis = (window.innerWidth / 2 - e.pageX) / 20;
    let yAxis = (window.innerHeight / 2 - e.pageY) / 20;
    heroText.style.transform = `translate(${xAxis}px, ${yAxis}px)`;
});

// Hiệu ứng hover cho các item trong song list
const songItems = document.querySelectorAll('.song-item');

songItems.forEach((item) => {
    item.addEventListener('mouseenter', () => {
        item.style.transform = 'scale(1.1)';
    });

    item.addEventListener('mouseleave', () => {
        item.style.transform = 'scale(1)';
    });
});

// // Tải tệp midi
// MIDI.Player.loadFile("../midi/melody_lstm_att.mid", function () {
//     // Khi tệp midi được tải, hiển thị bàn phím piano và phát tệp midi
//     MIDI.Player.start();
//     MIDI.visuals.keyboard(document.getElementById("keyboard"), MIDI.Player);
// });