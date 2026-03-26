# Muhammad's LeetCode Journal

A public coding journal tracking 150 LeetCode problems across 6 months — built for the Autumn 2026 AI internship hunt.

## Features
- **Public view** — anyone can see all solutions, notes, and progress stats
- **Admin login** — Firebase Auth (your account only) to add/edit solutions
- **Code editor** — paste code or upload `.py`, `.js`, `.java` etc files
- **150 problems pre-loaded** — organised by month and topic from the 6-month plan

## Setup

### 1. Firebase config
Go to [Firebase Console](https://console.firebase.google.com/) → `family-tree-cafc6` project:
- **Authentication** → Enable Email/Password provider → Add your email account
- **Firestore** → Create database → Set rules (see below)
- **Project Settings** → Copy your config into `src/firebase.js`

### Firestore Security Rules
```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /solutions/{docId} {
      allow read: if true;                          // public can read
      allow write: if request.auth != null;         // only logged-in admin can write
    }
  }
}
```

### 2. Install & run
```bash
npm install
npm run dev
```

### 3. Deploy to Vercel
```bash
npm install -g vercel
vercel --prod
```

Or push to GitHub and connect repo to Vercel — it auto-deploys.

## Project structure
```
src/
  data/problems.js      — all 150 problems pre-loaded
  firebase.js           — Firebase config (fill in your keys)
  pages/
    Journal.jsx         — public view
    Admin.jsx           — admin dashboard (protected)
    Login.jsx           — login page
  components/
    Navbar.jsx
    SolutionEditor.jsx  — code paste + file upload
  hooks/useAuth.jsx
```

## Usage
1. Visit `/` — public journal, anyone can read
2. Visit `/login` — sign in with your Firebase email/password
3. Visit `/admin` — select a problem, paste your solution, save
4. Solution immediately appears on the public `/` page
