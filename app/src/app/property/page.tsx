'use client';

import { Container, SimpleGrid } from '@mantine/core';
import Form from './Form';

export default function SalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      <Form />
    </Container>
  );
}
