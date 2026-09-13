import js from '@eslint/js'
import tsPlugin from '@typescript-eslint/eslint-plugin'
import tsParser from '@typescript-eslint/parser'
import perfectionist from 'eslint-plugin-perfectionist'
import unusedImports from 'eslint-plugin-unused-imports'

import base from '../eslint.base.mjs'

export default [
  {
    ignores: ['src/rpc/generated.ts', 'dist/**', 'node_modules/**', '**/*.config.*']
  },
  ...base({ js, tsPlugin, tsParser, unusedImports, perfectionist }),
  {
    files: ['src/**/*.ts'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'error'
    }
  },
  {
    files: ['src/__tests__/**'],
    rules: {
      '@typescript-eslint/no-explicit-any': 'off'
    }
  }
]
