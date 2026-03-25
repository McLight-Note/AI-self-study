import React, { useEffect, useState } from 'react'
import { collection, getDocs } from 'firebase/firestore'
import { signOut } from 'firebase/auth'
import { useNavigate, Link } from 'react-router-dom'
import { db, auth } from '../firebase'
import { PROBLEMS, MONTH_LABELS } from '../data/problems'

const DIFF_COLOR = { Easy:'#00ff88', Medium:'#ffb800', Hard:'#ff4444' }

export default function Dashboard() {
  const [solutions, setSolutions] = useState({})
  const [filterMonth, setFilterMonth] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const navigate = useNavigate()

  useEffect(() => {
    getDocs(collection(db, 'solutions')).then(snap => {
      const data = {}
      snap.forEach(doc => { data[doc.id] = doc.data() })
      setSolutions(data)
    })
  }, [])

  const solved = Object.values(solutions).filter(s => s.status === 'solved').length
  const inProgress = Object.values(solutions).filter(s => s.status === 'in-progress').length

  const filtered = PROBLEMS.filter(p => {
    if (filterMonth !== 'all' && p.month !== Number(filterMonth)) return false
    if (filterStatus === 'solved') return solutions[p.id]?.status === 'solved'
    if (filterStatus === 'in-progress') return solutions[p.id]?.status === 'in-progress'
    if (filterStatus === 'unsolved') return !solutions[p.id] || solutions[p.id]?.status === 'unsolved'
    return true
  })

  const grouped = filtered.reduce((acc, p) => {
    const key = `M${p.month}: ${MONTH_LABELS[p.month]}`
    if (!acc[key]) acc[key] = []
    acc[key].push(p)
    return acc
  }, {})

  return (
    <div style={s.page}>
      {/* Header */}
      <header style={s.header}>
        <div style={s.logo}><span style={{color:'#00ff88'}}>&gt;_</span> lc.journal <span style={s.adminBadge}>admin</span></div>
        <div style={s.headerRight}>
          <a href="/" style={s.navLink}>public view</a>
          <button style={s.logoutBtn} onClick={() => signOut(auth).then(() => navigate('/login'))}>logout</button>
        </div>
      </header>

      <div style={s.container}>
        {/* Stats */}
        <div style={s.statsRow}>
          <div style={s.stat}><div style={s.statNum}>{PROBLEMS.length}</div><div style={s.statLabel}>total</div></div>
          <div style={s.stat}><div style={{...s.statNum, color:'#00ff88'}}>{solved}</div><div style={s.statLabel}>solved</div></div>
          <div style={s.stat}><div style={{...s.statNum, color:'#ffb800'}}>{inProgress}</div><div style={s.statLabel}>in progress</div></div>
          <div style={s.stat}><div style={{...s.statNum, color:'#888'}}>{PROBLEMS.length - solved - inProgress}</div><div style={s.statLabel}>unsolved</div></div>
          <div style={s.progressBar}>
            <div style={{...s.progressFill, width:`${Math.round(solved/PROBLEMS.length*100)}%`}}/>
            <span style={s.progressLabel}>{Math.round(solved/PROBLEMS.length*100)}%</span>
          </div>
        </div>

        {/* Filters */}
        <div style={s.filters}>
          <select style={s.select} value={filterMonth} onChange={e => setFilterMonth(e.target.value)}>
            <option value="all">all months</option>
            {[1,2,3,4,5,6].map(m => <option key={m} value={m}>month {m}</option>)}
          </select>
          <select style={s.select} value={filterStatus} onChange={e => setFilterStatus(e.target.value)}>
            <option value="all">all status</option>
            <option value="solved">solved</option>
            <option value="in-progress">in progress</option>
            <option value="unsolved">unsolved</option>
          </select>
        </div>

        {/* Problem list grouped by month */}
        {Object.entries(grouped).map(([group, problems]) => (
          <div key={group} style={s.group}>
            <div style={s.groupHeader}>{group}</div>
            <div style={s.problemGrid}>
              {problems.map(p => {
                const sol = solutions[p.id]
                const status = sol?.status || 'unsolved'
                return (
                  <div key={p.id} style={s.problemCard}>
                    <div style={s.cardTop}>
                      <span style={{...s.diff, color: DIFF_COLOR[p.difficulty]}}>{p.difficulty}</span>
                      <span style={{...s.statusDot, background: status==='solved'?'#00ff88':status==='in-progress'?'#ffb800':'#333'}} title={status}/>
                    </div>
                    <div style={s.cardTitle}>{p.title}</div>
                    <div style={s.cardTopic}>{p.topic}</div>
                    <div style={s.cardActions}>
                      <Link to={`/edit/${p.id}`} style={s.editBtn}>
                        {status === 'unsolved' ? '+ add solution' : '✎ edit'}
                      </Link>
                      <a href={p.leetcodeUrl} target="_blank" rel="noopener noreferrer" style={s.lcLink}>LC ↗</a>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', background:'#0a0a0a', color:'#f0f0f0' },
  header: { display:'flex', alignItems:'center', justifyContent:'space-between', padding:'1rem 2rem', borderBottom:'1px solid #1a1a1a', position:'sticky', top:0, background:'#0a0a0a', zIndex:10 },
  logo: { fontFamily:"'Syne',sans-serif", fontSize:'1.2rem', fontWeight:800 },
  adminBadge: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.6rem', background:'#1a1a1a', border:'1px solid #2a2a2a', color:'#888', padding:'2px 6px', borderRadius:'4px', marginLeft:'0.5rem', verticalAlign:'middle' },
  headerRight: { display:'flex', gap:'1rem', alignItems:'center' },
  navLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#888', textDecoration:'none' },
  logoutBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', background:'none', border:'1px solid #2a2a2a', color:'#888', padding:'0.3rem 0.7rem', borderRadius:'4px', cursor:'pointer' },
  container: { maxWidth:'1100px', margin:'0 auto', padding:'2rem' },
  statsRow: { display:'flex', gap:'1rem', alignItems:'center', flexWrap:'wrap', marginBottom:'2rem', background:'#111', border:'1px solid #1a1a1a', borderRadius:'10px', padding:'1.2rem 1.5rem' },
  stat: { display:'flex', flexDirection:'column', alignItems:'center', minWidth:'60px' },
  statNum: { fontFamily:"'Syne',sans-serif", fontSize:'1.8rem', fontWeight:800, lineHeight:1 },
  statLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', marginTop:'2px' },
  progressBar: { flex:1, minWidth:'120px', height:'6px', background:'#1a1a1a', borderRadius:'3px', position:'relative', display:'flex', alignItems:'center' },
  progressFill: { height:'100%', background:'#00ff88', borderRadius:'3px', transition:'width 0.5s' },
  progressLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#00ff88', marginLeft:'0.5rem', position:'absolute', right:0 },
  filters: { display:'flex', gap:'0.75rem', marginBottom:'1.5rem' },
  select: { background:'#111', border:'1px solid #2a2a2a', borderRadius:'6px', color:'#888', padding:'0.5rem 0.8rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', cursor:'pointer' },
  group: { marginBottom:'2.5rem' },
  groupHeader: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', color:'#555', textTransform:'uppercase', letterSpacing:'0.1em', marginBottom:'0.75rem', paddingBottom:'0.5rem', borderBottom:'1px solid #1a1a1a' },
  problemGrid: { display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(220px, 1fr))', gap:'0.75rem' },
  problemCard: { background:'#111', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem', transition:'border-color 0.2s' },
  cardTop: { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'0.5rem' },
  diff: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', fontWeight:700 },
  statusDot: { width:'8px', height:'8px', borderRadius:'50%' },
  cardTitle: { fontFamily:"'Syne',sans-serif", fontSize:'0.85rem', fontWeight:600, marginBottom:'0.25rem', lineHeight:1.3 },
  cardTopic: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', marginBottom:'0.8rem' },
  cardActions: { display:'flex', gap:'0.5rem', alignItems:'center' },
  editBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#00ff88', textDecoration:'none', background:'#0a1a0f', border:'1px solid #00ff8833', padding:'0.25rem 0.5rem', borderRadius:'4px' },
  lcLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', textDecoration:'none' },
}
