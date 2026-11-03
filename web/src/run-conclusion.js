import { failureSummary } from './failure-report.js';

// Conclusions come from recorded reports or raw failures, never a reconstructed workflow.
export function runConclusion(run = {}, events = []) {
  if (['running', 'idle', 'succeeded'].includes(run.status) && !run.failure_report) return '';
  return failureSummary(run, events);
}
