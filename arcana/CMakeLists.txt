# Nebula library
add_library(Arcana INTERFACE)

# Core modules
add_subdirectory(nn)
add_subdirectory(loss)
add_subdirectory(tensor)
add_subdirectory(autograd)

# Link all modules
target_link_libraries(Arcana INTERFACE
    Arcana::Tensor
    Arcana::NN
    Arcana::Loss
    Arcana::Autograd
)