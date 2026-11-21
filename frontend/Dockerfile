# ---- build: exact dependency tree from package-lock.json ----
FROM node:24.22.2-alpine3.23 AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY . .
RUN npm run build

# ---- test (optional target: docker build --target test .) ----
FROM build AS test
RUN npm test

# ---- serve: static files + /api proxy ----
FROM nginx:1.29.8-alpine3.23
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
