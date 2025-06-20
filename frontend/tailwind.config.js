module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        handwriting: ['ZCOOL KuaiLe', 'cursive'],
        sans: ['Noto Sans TC', 'sans-serif'],
      },
      colors: {
        card: '#fffdfa',
        accent: '#f7c59f',
        doodle: '#b7d7c1',
      },
      borderRadius: {
        '3xl': '2rem',
        '4xl': '2.5rem',
      },
    },
  },
  plugins: [],
}