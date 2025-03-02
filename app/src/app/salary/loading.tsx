'use client';

import { Container, SimpleGrid } from '@mantine/core';
import SalaryFormSkeleton from './SalaryFormSkeleton';

export default function LoadingSalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      <SalaryFormSkeleton />
    </Container>
  );
}
