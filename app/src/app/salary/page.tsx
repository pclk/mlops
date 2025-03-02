'use client';

import { Container, SimpleGrid } from '@mantine/core';
import SalaryForm from './SalaryForm';

export default function SalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      <SalaryForm />
    </Container>
  );
}
