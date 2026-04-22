const BASE = '/api'

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  }
  if (body !== undefined) opts.body = JSON.stringify(body)
  const res = await fetch(`${BASE}${path}`, opts)
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${err}`)
  }
  return res.json()
}

// Unwrap openenv envelope: {observation: {...}, reward, done} → flat object with reward/done merged in
function unwrapObservation(envResponse) {
  if (envResponse && typeof envResponse.observation === 'object') {
    return { ...envResponse.observation, reward: envResponse.reward, done: envResponse.done }
  }
  return envResponse
}

// POST /reset — start a new episode (openenv wire format)
export async function resetEpisode() {
  const raw = await request('POST', '/reset', {})
  return unwrapObservation(raw)
}

// POST /step — advance episode with an action (openenv wire format)
// episode_id is an extra field on the StepRequest body (StepRequest has extra="allow")
export async function stepEpisode(episode_id, action) {
  const raw = await request('POST', '/step', { action, episode_id })
  return unwrapObservation(raw)
}

// GET /episode_state?episode_id= — inspect specific episode without advancing
export async function getState(episode_id) {
  return request('GET', `/episode_state?episode_id=${episode_id}`)
}

// GET /tasks — full task bank
export async function getTasks() {
  return request('GET', '/tasks')
}

// POST /grader — standalone offline grader
export async function runGrader(payload) {
  return request('POST', '/grader', payload)
}

// GET /baseline — pre-computed baseline scores
export async function getBaseline() {
  return request('GET', '/baseline')
}

// GET /forge/queue — active queue state
export async function getForgeQueue() {
  return request('GET', '/forge/queue')
}

// GET /forge/stats — aggregate Forge statistics
export async function getForgeStats() {
  return request('GET', '/forge/stats')
}

// GET /oversight/stats — detection/explanation/correction per domain+corruption
export async function getOversightStats() {
  return request('GET', '/oversight/stats')
}

// GET /oversight/difficulty_curve — pass@k time series
export async function getDifficultyCurve() {
  return request('GET', '/oversight/difficulty_curve')
}
