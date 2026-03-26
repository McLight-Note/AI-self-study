import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { AuthProvider, useAuth } from './hooks/useAuth'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import ProblemEditor from './pages/ProblemEditor'
import PublicView from './pages/PublicView'
import PublicProblem from './pages/PublicProblem'

function PrivateRoute({ children }) {
  const user = useAuth()
  if (user === undefined) return <div style={{display:'flex',alignItems:'center',justifyContent:'center',height:'100vh',color:'#00ff88',fontFamily:'JetBrains Mono'}}>loading...</div>
  return user ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/dashboard" element={<PrivateRoute><Dashboard /></PrivateRoute>} />
          <Route path="/edit/:problemId" element={<PrivateRoute><ProblemEditor /></PrivateRoute>} />
          <Route path="/" element={<PublicView />} />
          <Route path="/problem/:problemId" element={<PublicProblem />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  )
}
