'use client';

import { Container, SimpleGrid } from '@mantine/core';
import FormSkeleton from './FormSkeleton';

export default function LoadingSalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      <FormSkeleton />
    </Container>
  );
}
