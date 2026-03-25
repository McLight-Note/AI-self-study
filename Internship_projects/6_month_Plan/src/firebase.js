// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAmch_T5CsxvqlNOKCLcCDK-mtHUZ9XWSQ",
  authDomain: "development-month.firebaseapp.com",
  projectId: "development-month",
  storageBucket: "development-month.firebasestorage.app",
  messagingSenderId: "714714800614",
  appId: "1:714714800614:web:8e08378932dc9599d8b627",
  measurementId: "G-F8F0P5MDP3"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);