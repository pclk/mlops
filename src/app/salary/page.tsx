import { Container, SimpleGrid } from '@mantine/core';
import SalaryForm from './SalaryForm';
import Chat from './Chat';

export default function SalaryPage() {
  return (
    <Container size="xl" py="xl" style={{ height: '100svh' }}>
      <SimpleGrid cols={{ base: 1, md: 12 }} spacing="md" style={{ height: '100%' }}>
        <div style={{ gridColumn: 'span 4', height: '100%' }}>
          <Chat />
        </div>
        <div style={{ gridColumn: 'span 8', height: '100%' }}>
          <SalaryForm />
        </div>
      </SimpleGrid>
    </Container>
  );
}
