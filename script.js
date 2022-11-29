let model = {},
	data = [],
	currentIndex = -1,
	detectedFont = ''
const fontMap = {
	0: 'AGENCY',
	1: 'FREESTYLE',
	2: 'MODERN',
	3: 'ONYX',
	4: 'PRISTINA',
	5: 'CALISTO',
	6: 'CENTAUR',
	7: 'IMPRINT',
	8: 'INFORMAL',
	9: 'PALACE',
}
const canvas = document.querySelector('.char-display'),
	btnNewChar = document.querySelector('.btn-new-char'),
	btnPredict = document.querySelector('.btn-predict'),
	output = document.querySelector('.output'),
	outputGroup = document.querySelector('.output-group')
const context = canvas.getContext('2d')
const size = 20

const random = (n) => Math.floor(Math.random() * n)

const invertKeys = (obj) => Object.fromEntries(Object.entries(obj).map((entry) => entry.reverse()))

const displayChar = () => {
	const row = data[currentIndex]

	if (!row) {
		canvas.width += 0
		return
	}

	const intensities = row.slice(-(size ** 2))
	const imageData = context.createImageData(size, size)
	for (let i = 0; i < size; i++) {
		for (let j = 0; j < size; j++) {
			imageData.data[(i * size + j) * 4] =
				imageData.data[(i * size + j) * 4 + 1] =
				imageData.data[(i * size + j) * 4 + 2] =
					intensities[i * size + j]
			imageData.data[(i * size + j) * 4 + 3] = 255
		}
	}
	context.putImageData(imageData, 0, 0)
}

const updateDisplay = () => {
	displayChar()
}

const updateOutput = (str) => {
	detectedFont = str
	output.textContent = str
	str ? outputGroup.removeAttribute('hidden') : outputGroup.setAttribute('hidden', '')
}

const chooseNewChar = () => {
	currentIndex = random(data.length)
	updateDisplay()
	updateOutput('')
}

const preprocess = (row) => {
	const result = row
		.slice(0, -(size ** 2))
		.concat(row.slice(-(size ** 2)).map((intensity) => intensity / 255))
	result.splice(0, 2)
	return result
}

const predict = async () => {
	let row = data[currentIndex]
	if (!row) return
	row = preprocess(row)

	let scores = await model.predict([tf.tensor(row).reshape([1, row.length])]).array()
	scores = scores[0]
	const sorted = scores.map((a) => a).sort((a, b) => b - a)
	const predicted = scores.indexOf(sorted[sorted[1] > 0 ? 1 : 0])
	return fontMap[predicted]
}

const predictAll = async () => {
	const predictions = Array(10).fill(0)
	currentIndex = 0
	for (let i = 0; i < data.length; i++) {
		const predicted = await predict()
		predictions[predicted]++
		currentIndex++
	}
	console.log(predictions)
}

/* ----- Initialization ----- */

const init = async () => {
	model = await tf.loadLayersModel('model/model.json')

	const response = await fetch('data/test.csv')
	const csv = await response.text()
	data = csv
		.trim()
		.split('\n')
		.map((row) => row.split(',').map((value, i) => (i <= 1 ? value : +value)))

	canvas.width = canvas.height = 20

	btnNewChar.addEventListener('click', () => {
		chooseNewChar()
	})
	btnPredict.addEventListener('click', async () => {
		updateOutput('...')
		const font = await predict()
		updateOutput(font)
	})
}

init()
