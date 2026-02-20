// Конфигурация фона
const BACK_FOLDER = "static/img/back/";
const BACK_IMAGES = ["1.jpg", "2.jpg", "3.jpg"];

// Элементы
const frame = document.getElementById('frame');
const displayImg = document.getElementById('display-image');
const placeholder = document.getElementById('placeholder');
const chooseBtn = document.getElementById('choose-btn');
const fileInput = document.getElementById('file-input');

// Считываем время анимации прямо из CSS (переводим 2.0s в 2000ms)
const style = getComputedStyle(document.documentElement);
const fadeDuration = parseFloat(style.getPropertyValue('--fade-duration')) * 1000;

const sliderInput = document.getElementById('iterations-slider');
const scaleLabels = document.querySelectorAll('.scale-label');

let selectedIterations = 1;
let selectedFile = null;


// Установка фона
function setRandomBackground() {
    const randomImg = BACK_IMAGES[Math.floor(Math.random() * BACK_IMAGES.length)];
    document.body.style.backgroundImage = `url('${BACK_FOLDER}${randomImg}')`;
}

// Функция для обновления визуального состояния шкалы
function updateSliderVisuals(value) {
    scaleLabels.forEach(label => {
        // parseInt нужен, так как атрибуты и value инпута - это строки
        if (parseInt(label.dataset.value) === value) {
            label.classList.add('active');
        } else {
            label.classList.remove('active');
        }
    });
}

// Слушаем движение ползунка в реальном времени ('input')
sliderInput.addEventListener('input', (e) => {
    const newValue = parseInt(e.target.value);
    selectedIterations = newValue;
    updateSliderVisuals(newValue);
    // console.log("Выбрано итераций:", selectedIterations); // Для отладки
});

// Плавная смена картинки
function updateImage(src) {
    displayImg.classList.remove('fading-in');
    displayImg.classList.add('fading-out');

    setTimeout(() => {
        displayImg.src = src;
        displayImg.style.display = 'block';
        displayImg.style.transition = `opacity var(--fade-duration) ease-in-out`;
        placeholder.style.display = 'none';

        displayImg.onload = () => {
            displayImg.classList.replace('fading-out', 'fading-in');
        };
    }, 255); // Небольшой зазор для начала исчезновения
}

// Обработка выбора файла
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (ev) => updateImage(ev.target.result);
        reader.readAsDataURL(file);
        chooseBtn.textContent = 'DENOISE';
    }
});

// Отправка на сервер
async function sendToServer(file) {
    chooseBtn.disabled = true;
    frame.classList.add('processing'); // Включаем пульсацию рамки

    const fade = fadeDuration

    if (displayImg) {
        // Устанавливаем временный быстрый переход через JS
        displayImg.style.transition = `opacity ${fade}ms ease-in-out`;
        displayImg.classList.add('fading-out');

        await new Promise(resolve => setTimeout(resolve, fade));
    }

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('iterations', selectedIterations);

        chooseBtn.textContent = 'PROCESSING...';
        const response = await fetch('/predict', { method: 'POST', body: formData });

        if (!response.ok) throw new Error('Server error');

        const blob = await response.blob();
        updateImage(URL.createObjectURL(blob)); // Показываем результат

        chooseBtn.textContent = 'CHOOSE IMAGE';
        selectedFile = null;
    } catch (err) {
        console.error(err);
        chooseBtn.textContent = 'ERROR! TRY AGAIN';
    } finally {
        chooseBtn.disabled = false;
        frame.classList.remove('processing'); // Выключаем пульсацию
    }
}

// Обработчик кнопки
chooseBtn.addEventListener('click', () => {
if (!selectedFile) {
        fileInput.click();
    } else {
        sendToServer(selectedFile);
    }
});

window.addEventListener("load", setRandomBackground);