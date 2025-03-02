'use client';

import { Container, SimpleGrid } from '@mantine/core';
import SalaryFormSkeleton from './SalaryFormSkeleton';

export default function LoadingSalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      {/* <SimpleGrid cols={{ base: 1, md: 12 }} spacing="md" style={{ height: '100%' }}> */}
      {/*   <div style={{ gridColumn: 'span 4', height: '100%' }}> */}
      {/*     {/* Chat skeleton can go here if needed */}
      {/*   </div> */}
      {/*   <div style={{ gridColumn: 'span 8', height: '100%' }}> */}
      <SalaryFormSkeleton />
      {/*   </div> */}
      {/* </SimpleGrid> */}
    </Container>
  );
}
