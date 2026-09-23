import js from '@eslint/js';
import ts from 'typescript-eslint';
import globals from 'globals';
export default ts.config(
  { ignores: ['**/node_modules/**','**/.next/**','**/dist/**','**/next-env.d.ts','**/playwright-report/**','**/test-results/**','.pnpm-store/**','services/**','apps/web/public/mediapipe/**'] },
  js.configs.recommended,
  ...ts.configs.recommended,
  { files: ['**/*.{ts,tsx,mjs}'], languageOptions: { globals: {...globals.node,...globals.browser} }, rules: {
    '@typescript-eslint/no-explicit-any': 'off',
    '@typescript-eslint/no-unused-vars': ['error',{argsIgnorePattern:'^_',varsIgnorePattern:'^_',caughtErrors:'none'}],
    'no-empty': ['error',{allowEmptyCatch:true}],
  } },
);
