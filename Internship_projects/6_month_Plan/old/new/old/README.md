# 6-Month Goal Tracker

An interactive goal tracking application for specializing in AI Multi-Object Tracking and building a production-ready system.

## Features

- ✅ Track progress on technical and personal goals
- 📊 Visual progress indicators for each month
- 💾 Automatic progress saving (localStorage)
- 📱 Fully responsive design
- 🎨 Beautiful UI with Tailwind CSS

## Deploy to Vercel

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Push code to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/goal-tracker.git
   git push -u origin main
   ```

2. **Deploy on Vercel:**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect Vite settings
   - Click "Deploy"
   - Done! Your app will be live in ~2 minutes

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI:**
   ```bash
   npm i -g vercel
   ```

2. **Deploy:**
   ```bash
   vercel
   ```
   
3. **Follow the prompts:**
   - Login to your Vercel account
   - Confirm project settings
   - Deploy!

## Local Development

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Open browser:**
   Navigate to `http://localhost:5173`

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` folder.

## Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **localStorage** - Progress persistence

## Project Structure

```
goal-tracker/
├── src/
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── index.html           # HTML template
├── package.json         # Dependencies
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind configuration
└── README.md           # This file
```

## Customization

You can customize your goals by editing the `roadmap` array in `src/App.jsx`. Each month can have multiple categories with tasks, resources, and progress tracking.

## License

MIT
