program read_binary_file
  implicit none
  
  integer, parameter :: file_unit = 10
  integer, parameter :: num_elements = 10000
  integer, parameter :: output_unit = 20
  
  real :: data(num_elements)
  integer :: i
  
  ! Open the binary file for reading
  open(unit=file_unit, file='gridpop', access='stream', status='old', &
   form='unformatted', action='read', iostat=i)
  if (i /= 0) then
    write(*,*) "Error opening file:", i
    stop
  end if
  
  ! Read the data from the binary file
  read(file_unit) data
  
  ! Close the file
  close(file_unit)


  open(unit=output_unit, file='output.txt', status='replace')

  ! Process the data as needed
  do i = 1, num_elements
    write(output_unit,*) data(i)
  end do

  close(output_unit)

  
end program read_binary_file

