#!/bin/bash

echo "🚀 Goal Tracker - Quick Deploy to Vercel"
echo "========================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git add .
    git commit -m "Initial commit: Advanced MOT Goal Tracker"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

echo ""
echo "Next steps:"
echo "1. Create a GitHub repository at https://github.com/new"
echo "2. Run these commands to push your code:"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/goal-tracker.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Go to https://vercel.com and click 'New Project'"
echo "4. Import your GitHub repository"
echo "5. Click 'Deploy' - Vercel auto-detects Vite!"
echo ""
echo "🎉 Your app will be live in ~2 minutes!"
