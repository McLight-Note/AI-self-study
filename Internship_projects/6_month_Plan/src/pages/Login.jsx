import React, { useState } from 'react'
import { signInWithEmailAndPassword } from 'firebase/auth'
import { useNavigate } from 'react-router-dom'
import { auth } from '../firebase'

export default function Login() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleLogin = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await signInWithEmailAndPassword(auth, email, password)
      navigate('/dashboard')
    } catch (err) {
      setError('Invalid credentials. Check email/password.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={s.page}>
      <div style={s.card}>
        <div style={s.logo}>
          <span style={s.logoAccent}>&gt;_</span> lc.journal
        </div>
        <p style={s.sub}>admin access only</p>
        <form onSubmit={handleLogin} style={s.form}>
          <div style={s.field}>
            <label style={s.label}>email</label>
            <input style={s.input} type="email" value={email}
              onChange={e => setEmail(e.target.value)} placeholder="you@email.com" required />
          </div>
          <div style={s.field}>
            <label style={s.label}>password</label>
            <input style={s.input} type="password" value={password}
              onChange={e => setPassword(e.target.value)} placeholder="••••••••" required />
          </div>
          {error && <p style={s.error}>{error}</p>}
          <button style={{...s.btn, opacity: loading ? 0.6 : 1}} type="submit" disabled={loading}>
            {loading ? 'signing in...' : 'sign in →'}
          </button>
        </form>
        <a href="/" style={s.backLink}>← public view</a>
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', display:'flex', alignItems:'center', justifyContent:'center', background:'#0a0a0a', padding:'1rem' },
  card: { width:'100%', maxWidth:'380px', background:'#111', border:'1px solid #2a2a2a', borderRadius:'12px', padding:'2.5rem 2rem' },
  logo: { fontFamily:"'Syne',sans-serif", fontSize:'1.6rem', fontWeight:800, color:'#f0f0f0', marginBottom:'0.25rem' },
  logoAccent: { color:'#00ff88' },
  sub: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#555', marginBottom:'2rem' },
  form: { display:'flex', flexDirection:'column', gap:'1.2rem' },
  field: { display:'flex', flexDirection:'column', gap:'0.4rem' },
  label: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', color:'#888', textTransform:'uppercase', letterSpacing:'0.1em' },
  input: { background:'#0a0a0a', border:'1px solid #2a2a2a', borderRadius:'6px', padding:'0.7rem 0.9rem', color:'#f0f0f0', fontSize:'0.9rem', outline:'none', fontFamily:"'JetBrains Mono',monospace" },
  error: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#ff4444' },
  btn: { background:'#00ff88', color:'#0a0a0a', border:'none', borderRadius:'6px', padding:'0.8rem', fontWeight:700, fontSize:'0.9rem', fontFamily:"'Syne',sans-serif", cursor:'pointer', transition:'opacity 0.2s' },
  backLink: { display:'block', textAlign:'center', marginTop:'1.5rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#555', textDecoration:'none' }
}
