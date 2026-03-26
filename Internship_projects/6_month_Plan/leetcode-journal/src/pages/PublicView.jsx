import React, { useEffect, useState } from 'react'
import { collection, getDocs } from 'firebase/firestore'
import { Link } from 'react-router-dom'
import { db } from '../firebase'
import { PROBLEMS, MONTH_LABELS } from '../data/problems'

const DIFF_COLOR = { Easy:'#00ff88', Medium:'#ffb800', Hard:'#ff4444' }

export default function PublicView() {
  const [solutions, setSolutions] = useState({})
  const [filterMonth, setFilterMonth] = useState('all')
  const [filterTopic, setFilterTopic] = useState('all')
  const [filterDiff, setFilterDiff] = useState('all')
  const [filterStatus, setFilterStatus] = useState('all')
  const [search, setSearch] = useState('')

  useEffect(() => {
    getDocs(collection(db, 'solutions')).then(snap => {
      const data = {}
      snap.forEach(doc => { data[doc.id] = doc.data() })
      setSolutions(data)
    })
  }, [])

  const solved = Object.values(solutions).filter(s => s.status === 'solved').length
  const topics = [...new Set(PROBLEMS.map(p => p.topic))]

  const filtered = PROBLEMS.filter(p => {
    if (filterMonth !== 'all' && p.month !== Number(filterMonth)) return false
    if (filterTopic !== 'all' && p.topic !== filterTopic) return false
    if (filterDiff !== 'all' && p.difficulty !== filterDiff) return false
    if (filterStatus === 'solved' && solutions[p.id]?.status !== 'solved') return false
    if (filterStatus === 'unsolved' && solutions[p.id]?.status === 'solved') return false
    if (search && !p.title.toLowerCase().includes(search.toLowerCase())) return false
    return true
  })

  const grouped = filtered.reduce((acc, p) => {
    const key = p.month
    if (!acc[key]) acc[key] = []
    acc[key].push(p)
    return acc
  }, {})

  const easyTotal = PROBLEMS.filter(p=>p.difficulty==='Easy').length
  const medTotal = PROBLEMS.filter(p=>p.difficulty==='Medium').length
  const hardTotal = PROBLEMS.filter(p=>p.difficulty==='Hard').length
  const easySolved = PROBLEMS.filter(p=>p.difficulty==='Easy' && solutions[p.id]?.status==='solved').length
  const medSolved = PROBLEMS.filter(p=>p.difficulty==='Medium' && solutions[p.id]?.status==='solved').length
  const hardSolved = PROBLEMS.filter(p=>p.difficulty==='Hard' && solutions[p.id]?.status==='solved').length

  return (
    <div style={s.page}>
      {/* Hero header */}
      <header style={s.hero}>
        <div style={s.heroInner}>
          <div style={s.heroLeft}>
            <div style={s.heroLogo}><span style={{color:'#00ff88'}}>&gt;_</span> Muhammad's LeetCode Journal</div>
            <p style={s.heroSub}>6-month AI internship prep · 150 problems · Autumn 2026</p>
          </div>
          <Link to="/login" style={s.loginLink}>admin →</Link>
        </div>
      </header>

      <div style={s.container}>
        {/* Stats */}
        <div style={s.statsGrid}>
          <div style={s.bigStat}>
            <div style={s.bigNum}>{solved}<span style={s.bigDen}>/{PROBLEMS.length}</span></div>
            <div style={s.bigLabel}>problems solved</div>
            <div style={s.bigBar}><div style={{...s.bigBarFill, width:`${Math.round(solved/PROBLEMS.length*100)}%`}}/></div>
          </div>
          <div style={s.diffStats}>
            {[['Easy', easySolved, easyTotal, '#00ff88'], ['Medium', medSolved, medTotal, '#ffb800'], ['Hard', hardSolved, hardTotal, '#ff4444']].map(([d,sol,tot,col]) => (
              <div key={d} style={s.diffStat}>
                <div style={{...s.diffLabel, color:col}}>{d}</div>
                <div style={s.diffCount}>{sol}/{tot}</div>
                <div style={s.diffBar}><div style={{height:'100%',background:col,borderRadius:'2px',width:`${Math.round(sol/tot*100)}%`,transition:'width 0.5s'}}/></div>
              </div>
            ))}
          </div>
          <div style={s.monthStats}>
            {[1,2,3,4,5,6].map(m => {
              const mProbs = PROBLEMS.filter(p=>p.month===m)
              const mSolved = mProbs.filter(p=>solutions[p.id]?.status==='solved').length
              const pct = Math.round(mSolved/mProbs.length*100)
              return (
                <div key={m} style={s.monthStat}>
                  <div style={s.monthLabel}>M{m}</div>
                  <div style={s.monthBar}><div style={{height:'100%',background:`hsl(${pct*1.2},80%,50%)`,borderRadius:'2px',width:`${pct}%`,transition:'width 0.5s'}}/></div>
                  <div style={s.monthPct}>{mSolved}/{mProbs.length}</div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Filters */}
        <div style={s.filters}>
          <input style={s.search} placeholder="search problems..." value={search} onChange={e=>setSearch(e.target.value)} />
          <select style={s.select} value={filterMonth} onChange={e=>setFilterMonth(e.target.value)}>
            <option value="all">all months</option>
            {[1,2,3,4,5,6].map(m=><option key={m} value={m}>month {m}</option>)}
          </select>
          <select style={s.select} value={filterTopic} onChange={e=>setFilterTopic(e.target.value)}>
            <option value="all">all topics</option>
            {topics.map(t=><option key={t} value={t}>{t}</option>)}
          </select>
          <select style={s.select} value={filterDiff} onChange={e=>setFilterDiff(e.target.value)}>
            <option value="all">all difficulties</option>
            <option value="Easy">Easy</option>
            <option value="Medium">Medium</option>
            <option value="Hard">Hard</option>
          </select>
          <select style={s.select} value={filterStatus} onChange={e=>setFilterStatus(e.target.value)}>
            <option value="all">all status</option>
            <option value="solved">solved only</option>
            <option value="unsolved">unsolved only</option>
          </select>
        </div>

        {/* Problems by month */}
        {Object.entries(grouped).sort((a,b)=>Number(a[0])-Number(b[0])).map(([month, problems]) => {
          const mSolved = problems.filter(p=>solutions[p.id]?.status==='solved').length
          return (
            <div key={month} style={s.monthGroup}>
              <div style={s.monthHeader}>
                <div>
                  <span style={s.monthTag}>Month {month}</span>
                  <span style={s.monthTitle}>{MONTH_LABELS[Number(month)]}</span>
                </div>
                <span style={s.monthCount}>{mSolved}/{problems.length} solved</span>
              </div>
              <div style={s.problemGrid}>
                {problems.map(p => {
                  const sol = solutions[p.id]
                  const status = sol?.status || 'unsolved'
                  const hasSolution = status === 'solved' || status === 'in-progress'
                  return (
                    <div key={p.id} style={{...s.card, borderColor: status==='solved'?'#00ff8820':status==='in-progress'?'#ffb80020':'#1a1a1a'}}>
                      <div style={s.cardTop}>
                        <span style={{...s.diff, color:DIFF_COLOR[p.difficulty]}}>{p.difficulty}</span>
                        <span style={{...s.statusPill, background: status==='solved'?'#00ff8818':status==='in-progress'?'#ffb80018':'transparent', color: status==='solved'?'#00ff88':status==='in-progress'?'#ffb800':'#333'}}>
                          {status}
                        </span>
                      </div>
                      <div style={s.cardTitle}>{p.title}</div>
                      <div style={s.cardTopic}>{p.topic}</div>
                      <div style={s.cardBottom}>
                        {hasSolution
                          ? <Link to={`/problem/${p.id}`} style={s.viewBtn}>view solution →</Link>
                          : <span style={s.noSolution}>no solution yet</span>
                        }
                        <a href={p.leetcodeUrl} target="_blank" rel="noopener noreferrer" style={s.lcLink}>LC ↗</a>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', background:'#0a0a0a', color:'#f0f0f0' },
  hero: { borderBottom:'1px solid #1a1a1a', padding:'1.5rem 2rem' },
  heroInner: { maxWidth:'1100px', margin:'0 auto', display:'flex', justifyContent:'space-between', alignItems:'center' },
  heroLeft: {},
  heroLogo: { fontFamily:"'Syne',sans-serif", fontSize:'1.4rem', fontWeight:800, marginBottom:'0.3rem' },
  heroSub: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.72rem', color:'#555' },
  loginLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.72rem', color:'#333', textDecoration:'none', border:'1px solid #222', padding:'0.3rem 0.7rem', borderRadius:'4px' },
  container: { maxWidth:'1100px', margin:'0 auto', padding:'2rem' },
  statsGrid: { display:'grid', gridTemplateColumns:'auto 1fr 1fr', gap:'1.5rem', marginBottom:'2rem', background:'#111', border:'1px solid #1a1a1a', borderRadius:'12px', padding:'1.5rem' },
  bigStat: { paddingRight:'1.5rem', borderRight:'1px solid #1a1a1a' },
  bigNum: { fontFamily:"'Syne',sans-serif", fontSize:'3rem', fontWeight:800, color:'#00ff88', lineHeight:1 },
  bigDen: { fontSize:'1.5rem', color:'#333' },
  bigLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', margin:'0.3rem 0 0.75rem' },
  bigBar: { width:'140px', height:'4px', background:'#1a1a1a', borderRadius:'2px', overflow:'hidden' },
  bigBarFill: { height:'100%', background:'#00ff88', borderRadius:'2px', transition:'width 0.5s' },
  diffStats: { display:'flex', flexDirection:'column', justifyContent:'space-around', gap:'0.5rem' },
  diffStat: { display:'grid', gridTemplateColumns:'70px 40px 1fr', alignItems:'center', gap:'0.5rem' },
  diffLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', fontWeight:700 },
  diffCount: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', color:'#555', textAlign:'right' },
  diffBar: { height:'4px', background:'#1a1a1a', borderRadius:'2px', overflow:'hidden' },
  monthStats: { display:'flex', flexDirection:'column', justifyContent:'space-around', gap:'0.4rem' },
  monthStat: { display:'grid', gridTemplateColumns:'28px 1fr 40px', alignItems:'center', gap:'0.5rem' },
  monthLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555' },
  monthBar: { height:'4px', background:'#1a1a1a', borderRadius:'2px', overflow:'hidden' },
  monthPct: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.6rem', color:'#555', textAlign:'right' },
  filters: { display:'flex', gap:'0.6rem', flexWrap:'wrap', marginBottom:'1.75rem' },
  search: { background:'#111', border:'1px solid #2a2a2a', borderRadius:'6px', color:'#f0f0f0', padding:'0.45rem 0.8rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', outline:'none', minWidth:'180px' },
  select: { background:'#111', border:'1px solid #2a2a2a', borderRadius:'6px', color:'#888', padding:'0.45rem 0.7rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.72rem', cursor:'pointer' },
  monthGroup: { marginBottom:'2.5rem' },
  monthHeader: { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'0.75rem', paddingBottom:'0.5rem', borderBottom:'1px solid #1a1a1a' },
  monthTag: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#00ff88', background:'#00ff8812', padding:'2px 8px', borderRadius:'4px', marginRight:'0.75rem' },
  monthTitle: { fontFamily:"'Syne',sans-serif", fontSize:'0.9rem', fontWeight:600, color:'#888' },
  monthCount: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555' },
  problemGrid: { display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(210px, 1fr))', gap:'0.7rem' },
  card: { background:'#111', border:'1px solid', borderRadius:'8px', padding:'1rem', transition:'border-color 0.2s' },
  cardTop: { display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'0.4rem' },
  diff: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', fontWeight:700 },
  statusPill: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.6rem', padding:'2px 6px', borderRadius:'4px' },
  cardTitle: { fontFamily:"'Syne',sans-serif", fontSize:'0.85rem', fontWeight:600, marginBottom:'0.2rem', lineHeight:1.3 },
  cardTopic: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.62rem', color:'#555', marginBottom:'0.75rem' },
  cardBottom: { display:'flex', justifyContent:'space-between', alignItems:'center' },
  viewBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#00ff88', textDecoration:'none' },
  noSolution: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.62rem', color:'#333' },
  lcLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.62rem', color:'#444', textDecoration:'none' },
}
