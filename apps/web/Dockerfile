# Build context is the repo root (see docker-compose.yml) so the pnpm
# workspace (packages/types, packages/validation, ...) resolves correctly.
FROM node:20-slim AS builder
WORKDIR /repo
RUN corepack enable
COPY pnpm-workspace.yaml package.json pnpm-lock.yaml ./
COPY packages ./packages
COPY apps/web ./apps/web
COPY tooling ./tooling
RUN pnpm install --frozen-lockfile
RUN pnpm --filter @cropmerge/types build && pnpm --filter @cropmerge/validation build
RUN pnpm --filter @cropmerge/web build

FROM node:20-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /repo/apps/web/.output ./.output
EXPOSE 3000
CMD ["node", ".output/server/index.mjs"]
