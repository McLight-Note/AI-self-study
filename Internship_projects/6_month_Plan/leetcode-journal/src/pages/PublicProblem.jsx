import React, { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { doc, getDoc } from 'firebase/firestore'
import { db } from '../firebase'
import { PROBLEMS } from '../data/problems'

const DIFF_COLOR = { Easy:'#00ff88', Medium:'#ffb800', Hard:'#ff4444' }

export default function PublicProblem() {
  const { problemId } = useParams()
  const problem = PROBLEMS.find(p => p.id === problemId)
  const [sol, setSol] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDoc(doc(db, 'solutions', problemId)).then(d => {
      if (d.exists()) setSol(d.data())
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [problemId])

  if (!problem) return <div style={{padding:'2rem',color:'#ff4444',fontFamily:'JetBrains Mono'}}>Problem not found</div>
  if (loading) return <div style={{padding:'2rem',color:'#555',fontFamily:'JetBrains Mono,monospace'}}>loading...</div>

  const idx = PROBLEMS.findIndex(p => p.id === problemId)
  const prev = PROBLEMS[idx - 1]
  const next = PROBLEMS[idx + 1]

  return (
    <div style={s.page}>
      <header style={s.header}>
        <Link to="/" style={s.back}>← all problems</Link>
        <div style={s.headerMid}>
          <span style={{...s.diff, color:DIFF_COLOR[problem.difficulty]}}>{problem.difficulty}</span>
          <h1 style={s.title}>{problem.title}</h1>
          <span style={s.topic}>{problem.topic} · Month {problem.month}</span>
        </div>
        <a href={problem.leetcodeUrl} target="_blank" rel="noopener noreferrer" style={s.lcLink}>LeetCode ↗</a>
      </header>

      <div style={s.container}>
        {!sol || sol.status === 'unsolved' ? (
          <div style={s.noSol}>
            <div style={s.noSolIcon}>{'{ }'}</div>
            <p style={s.noSolText}>No solution posted yet.</p>
            <a href={problem.leetcodeUrl} target="_blank" rel="noopener noreferrer" style={s.lcBtn}>Try it on LeetCode ↗</a>
          </div>
        ) : (
          <div style={s.content}>
            {/* Meta bar */}
            <div style={s.metaBar}>
              {sol.status && <div style={s.metaItem}><span style={s.metaKey}>status</span><span style={{...s.metaVal, color: sol.status==='solved'?'#00ff88':'#ffb800'}}>{sol.status}</span></div>}
              {sol.language && <div style={s.metaItem}><span style={s.metaKey}>language</span><span style={s.metaVal}>{sol.language}</span></div>}
              {sol.timeComplexity && <div style={s.metaItem}><span style={s.metaKey}>time</span><span style={s.metaVal}>{sol.timeComplexity}</span></div>}
              {sol.spaceComplexity && <div style={s.metaItem}><span style={s.metaKey}>space</span><span style={s.metaVal}>{sol.spaceComplexity}</span></div>}
            </div>

            <div style={s.cols}>
              {/* Code */}
              <div style={s.col}>
                <div style={s.sectionLabel}>solution</div>
                {sol.code
                  ? <pre style={s.codeBlock}><code>{sol.code}</code></pre>
                  : <p style={s.empty}>no code added yet</p>}
              </div>

              {/* Notes */}
              <div style={s.col}>
                {sol.approach && <>
                  <div style={s.sectionLabel}>approach</div>
                  <div style={s.noteBlock}>{sol.approach}</div>
                </>}
                {sol.notes && <>
                  <div style={{...s.sectionLabel, marginTop:'1.5rem'}}>notes & learnings</div>
                  <div style={s.noteBlock}>{sol.notes}</div>
                </>}
                {!sol.approach && !sol.notes && <p style={s.empty}>no notes added yet</p>}
              </div>
            </div>
          </div>
        )}

        {/* Prev / Next */}
        <div style={s.nav}>
          {prev ? <Link to={`/problem/${prev.id}`} style={s.navBtn}>← {prev.title}</Link> : <span/>}
          {next ? <Link to={`/problem/${next.id}`} style={{...s.navBtn, textAlign:'right'}}>→ {next.title}</Link> : <span/>}
        </div>
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', background:'#0a0a0a', color:'#f0f0f0' },
  header: { display:'flex', alignItems:'center', justifyContent:'space-between', padding:'1rem 2rem', borderBottom:'1px solid #1a1a1a', position:'sticky', top:0, background:'#0a0a0a', zIndex:10, flexWrap:'wrap', gap:'0.5rem' },
  back: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#555', textDecoration:'none' },
  headerMid: { display:'flex', alignItems:'center', gap:'0.75rem', flex:1, justifyContent:'center', flexWrap:'wrap' },
  diff: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', fontWeight:700 },
  title: { fontFamily:"'Syne',sans-serif", fontSize:'1.1rem', fontWeight:700 },
  topic: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555' },
  lcLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.72rem', color:'#555', textDecoration:'none' },
  container: { maxWidth:'1100px', margin:'0 auto', padding:'2rem' },
  noSol: { display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'40vh', gap:'1rem' },
  noSolIcon: { fontFamily:"'JetBrains Mono',monospace", fontSize:'3rem', color:'#222' },
  noSolText: { fontFamily:"'Syne',sans-serif", fontSize:'1rem', color:'#555' },
  lcBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.8rem', color:'#00ff88', border:'1px solid #00ff8833', padding:'0.5rem 1.2rem', borderRadius:'6px', textDecoration:'none' },
  content: {},
  metaBar: { display:'flex', gap:'1.5rem', flexWrap:'wrap', background:'#111', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'0.9rem 1.2rem', marginBottom:'1.5rem' },
  metaItem: { display:'flex', gap:'0.5rem', alignItems:'center' },
  metaKey: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', textTransform:'uppercase' },
  metaVal: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#f0f0f0' },
  cols: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem', marginBottom:'2rem' },
  col: { display:'flex', flexDirection:'column' },
  sectionLabel: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', textTransform:'uppercase', letterSpacing:'0.1em', marginBottom:'0.5rem' },
  codeBlock: { background:'#0d0d0d', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1.25rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.82rem', lineHeight:1.7, overflowX:'auto', color:'#f0f0f0', whiteSpace:'pre' },
  noteBlock: { background:'#0d0d0d', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1.25rem', fontFamily:"'Syne',sans-serif", fontSize:'0.88rem', lineHeight:1.75, color:'#ccc', whiteSpace:'pre-wrap' },
  empty: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#333', fontStyle:'italic' },
  nav: { display:'flex', justifyContent:'space-between', borderTop:'1px solid #1a1a1a', paddingTop:'1.5rem' },
  navBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.72rem', color:'#555', textDecoration:'none', maxWidth:'40%', lineHeight:1.4 },
}
