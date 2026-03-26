import { useEffect, useState } from 'react';
import { collection, getDocs, query, orderBy } from 'firebase/firestore';
import { db } from '../firebase';
import { PROBLEMS, MONTH_NAMES } from '../data/problems';

const diffColor = { Easy: 'var(--easy)', Medium: 'var(--medium)', Hard: 'var(--hard)' };
const statusIcon = { solved: '✓', attempted: '~', 'needs-review': '?' };
const statusColor = { solved: 'var(--easy)', attempted: 'var(--medium)', 'needs-review': 'var(--red)' };

function StatCard({ label, value, sub }) {
  return (
    <div className="card" style={{ textAlign: 'center', padding: '20px 16px' }}>
      <div style={{ fontSize: 32, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent2)' }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: 'var(--text2)', marginTop: 4 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

export default function Journal() {
  const [solutions, setSolutions] = useState({});
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [selectedMonth, setSelectedMonth] = useState('all');
  const [expanded, setExpanded] = useState(null);

  useEffect(() => {
    const load = async () => {
      try {
        const snap = await getDocs(collection(db, 'solutions'));
        const map = {};
        snap.forEach(d => { map[d.id] = d.data(); });
        setSolutions(map);
      } catch (e) { console.error(e); }
      finally { setLoading(false); }
    };
    load();
  }, []);

  const solved = Object.values(solutions).filter(s => s.status === 'solved').length;
  const attempted = Object.values(solutions).filter(s => s.status === 'attempted').length;
  const easyDone = PROBLEMS.filter(p => p.difficulty === 'Easy' && solutions[p.id]).length;
  const medDone = PROBLEMS.filter(p => p.difficulty === 'Medium' && solutions[p.id]).length;
  const hardDone = PROBLEMS.filter(p => p.difficulty === 'Hard' && solutions[p.id]).length;

  const visible = PROBLEMS.filter(p => {
    if (selectedMonth !== 'all' && p.month !== Number(selectedMonth)) return false;
    if (filter === 'solved' && solutions[p.id]?.status !== 'solved') return false;
    if (filter === 'unsolved' && solutions[p.id]) return false;
    return true;
  });

  const months = [1, 2, 3, 4, 5, 6];

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '32px 20px' }}>

      {/* Hero */}
      <div className="fade-up" style={{ marginBottom: 40 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%', background: 'var(--easy)',
            animation: 'pulse-dot 2s infinite'
          }} />
          <span style={{ fontSize: 12, color: 'var(--easy)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
            ACTIVE — Autumn 2026 Internship Goal
          </span>
        </div>
        <h1 style={{ fontSize: 36, fontWeight: 700, letterSpacing: '-0.03em', lineHeight: 1.2 }}>
          Muhammad's<br />
          <span style={{ color: 'var(--accent2)' }}>LeetCode Journal</span>
        </h1>
        <p style={{ color: 'var(--text2)', marginTop: 12, maxWidth: 520, fontSize: 14 }}>
          150 problems across 6 months. Every solution, note, and pattern documented — publicly.
          Multi-Object Tracking AI internship target.
        </p>
      </div>

      {/* Stats */}
      <div className="fade-up" style={{
        display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
        gap: 12, marginBottom: 40
      }}>
        <StatCard label="Total Solved" value={solved} sub={`of 150`} />
        <StatCard label="Attempted" value={attempted} />
        <StatCard label="Easy ✓" value={easyDone} sub="of 29" />
        <StatCard label="Medium ✓" value={medDone} sub="of 93" />
        <StatCard label="Hard ✓" value={hardDone} sub="of 28" />
        <StatCard label="Progress" value={`${Math.round(solved / 150 * 100)}%`} />
      </div>

      {/* Progress bar per month */}
      <div className="card fade-up" style={{ marginBottom: 32, padding: 24 }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text2)', marginBottom: 16, letterSpacing: '0.06em' }}>
          MONTHLY PROGRESS
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {months.map(m => {
            const mProbs = PROBLEMS.filter(p => p.month === m);
            const mDone = mProbs.filter(p => solutions[p.id]?.status === 'solved').length;
            const pct = Math.round(mDone / mProbs.length * 100);
            return (
              <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text3)', width: 24 }}>
                  M{m}
                </span>
                <div style={{ flex: 1, height: 6, background: 'var(--bg3)', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', borderRadius: 3, width: `${pct}%`,
                    background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
                    transition: 'width 0.6s ease'
                  }} />
                </div>
                <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)', width: 60, textAlign: 'right' }}>
                  {mDone}/{mProbs.length}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {['all', 'solved', 'unsolved'].map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            style={{
              padding: '6px 14px', borderRadius: 6, fontSize: 13,
              background: filter === f ? 'var(--accent)' : 'var(--surface)',
              color: filter === f ? '#fff' : 'var(--text2)',
              border: `1px solid ${filter === f ? 'transparent' : 'var(--border)'}`,
              transition: 'all 0.15s'
            }}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
        <div style={{ marginLeft: 'auto' }}>
          <select
            value={selectedMonth}
            onChange={e => setSelectedMonth(e.target.value)}
            style={{ width: 'auto', fontSize: 13, padding: '6px 12px' }}
          >
            <option value="all">All months</option>
            {months.map(m => <option key={m} value={m}>Month {m}</option>)}
          </select>
        </div>
      </div>

      {/* Problem list grouped by topic */}
      {loading ? (
        <div style={{ textAlign: 'center', color: 'var(--text3)', padding: 60 }}>Loading...</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {visible.map((p, i) => {
            const sol = solutions[p.id];
            const isOpen = expanded === p.id;
            return (
              <div key={p.id} className="fade-up" style={{ animationDelay: `${i * 0.02}s` }}>
                <div
                  onClick={() => sol && setExpanded(isOpen ? null : p.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 12,
                    padding: '14px 18px',
                    background: 'var(--surface)',
                    border: `1px solid ${isOpen ? 'var(--accent)' : 'var(--border)'}`,
                    borderRadius: isOpen ? '12px 12px 0 0' : 12,
                    cursor: sol ? 'pointer' : 'default',
                    transition: 'border-color 0.2s',
                  }}
                >
                  {/* Status indicator */}
                  <div style={{
                    width: 28, height: 28, borderRadius: 8, flexShrink: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: sol ? 'rgba(52,211,153,0.1)' : 'var(--bg3)',
                    color: sol ? statusColor[sol.status] : 'var(--text3)',
                    fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)'
                  }}>
                    {sol ? statusIcon[sol.status] : '·'}
                  </div>

                  {/* Problem title */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: 500, fontSize: 14 }}>{p.title}</div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', marginTop: 2 }}>
                      M{p.month} · {p.topic}
                      {sol?.language && ` · ${sol.language}`}
                    </div>
                  </div>

                  {/* Complexity */}
                  {sol?.timeComplexity && (
                    <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text3)' }}>
                      {sol.timeComplexity}
                    </span>
                  )}

                  {/* Difficulty badge */}
                  <span className={`badge badge-${p.difficulty.toLowerCase()}`}>
                    {p.difficulty}
                  </span>

                  {/* LC link */}
                  <a
                    href={p.leetcodeUrl}
                    target="_blank"
                    rel="noreferrer"
                    onClick={e => e.stopPropagation()}
                    style={{ fontSize: 11, color: 'var(--text3)', padding: '4px 8px', borderRadius: 4, background: 'var(--bg3)' }}
                  >
                    LC ↗
                  </a>
                </div>

                {/* Expanded solution view */}
                {isOpen && sol && (
                  <div style={{
                    background: 'var(--bg2)',
                    border: '1px solid var(--accent)',
                    borderTop: 'none',
                    borderRadius: '0 0 12px 12px',
                    padding: 20,
                  }}>
                    {sol.notes && (
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.06em', marginBottom: 8 }}>
                          NOTES
                        </div>
                        <p style={{ fontSize: 14, color: 'var(--text2)', lineHeight: 1.7, whiteSpace: 'pre-wrap' }}>
                          {sol.notes}
                        </p>
                      </div>
                    )}
                    {sol.code && (
                      <div>
                        <div style={{
                          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                          marginBottom: 8
                        }}>
                          <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', letterSpacing: '0.06em' }}>
                            CODE — {sol.language?.toUpperCase()}
                          </span>
                          <div style={{ display: 'flex', gap: 12, fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
                            {sol.timeComplexity && <span>Time: {sol.timeComplexity}</span>}
                            {sol.spaceComplexity && <span>Space: {sol.spaceComplexity}</span>}
                          </div>
                        </div>
                        <pre style={{
                          background: 'var(--bg3)', borderRadius: 8, padding: 16,
                          fontFamily: 'var(--font-mono)', fontSize: 13, lineHeight: 1.7,
                          overflowX: 'auto', color: 'var(--text)',
                          border: '1px solid var(--border)'
                        }}>
                          <code>{sol.code}</code>
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
