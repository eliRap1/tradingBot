# Trading Bot — Claude Instructions

## Financial Logic

When implementing financial/trading logic (AMM pricing, P&L calculations, payouts), always write unit tests for edge cases before marking the feature complete. Double-check multipliers, minting logic, and balance flows.

## Deployment

Before suggesting any cloud service or deployment approach, verify the user's plan/tier supports it. For Firebase, assume the free Spark plan unless told otherwise.

## Testing

After implementing a feature that affects balances, credits, or visible state, always verify the change is reflected in the UI by checking for caching layers, stale data, or deploy propagation issues.

## Shell Commands

When running shell commands, use platform-appropriate syntax. Avoid bash-specific escaping in PowerShell. Always check the current shell before running diagnostic commands.
